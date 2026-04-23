"""
Entraînement ResNet18 sur les mammographies RSNA.

Classification au niveau image (chaque vue = un échantillon),
split stratifié au niveau exam pour éviter la fuite de données.

Usage:
    python -m fine_tuning.train_resnet
    python -m fine_tuning.train_resnet --epochs 100 --batch-size 4
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

from fine_tuning.config import (
    DEVICE,
    EXAM_LIST_PATH,
    IMAGE_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    RANDOM_SEED,
    WEIGHT_DECAY,
)
from fine_tuning.dataset import load_and_split

# ─── Hyperparamètres ResNet ──────────────────────────────────────────────────

IMG_SIZE = 512       # redimension côté court puis crop carré
BATCH_SIZE = 24       # images individuelles (pas des exams), 4 Go VRAM ok
CHECKPOINT_DIR  = Path(__file__).parent / "checkpoints"
RUNS_DIR        = CHECKPOINT_DIR / "runs"
EARLY_STOP_PATIENCE = 10
WARMUP_EPOCHS   = 5     # warm-up linéaire avant le cosine annealing
LR_PRETRAINED   = 1e-5  # LR par défaut quand on fine-tune depuis ImageNet
VIEWS = ["L-CC", "L-MLO", "R-CC", "R-MLO"]


def _make_run_dir(tag: str) -> Path:
    """Crée un dossier horodaté unique pour ce run."""
    run_dir = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{tag}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ─── Dataset image-level ─────────────────────────────────────────────────────

def _make_entries(exam_list: list) -> list[tuple[str, int]]:
    """Aplatit la liste d'exams en paires (chemin_image, label)."""
    entries = []
    for exam in exam_list:
        label = int(exam["cancer_label"]["malignant"])
        for view in VIEWS:
            for rel_path in exam.get(view, []):
                p = os.path.join(IMAGE_DIR, rel_path + ".png")
                if os.path.exists(p):
                    entries.append((p, label))
    return entries


class ImageDataset(Dataset):
    def __init__(
        self,
        entries: list[tuple[str, int]],
        mean: list[float],
        std: list[float],
        augment: bool = False,
        img_size: int = IMG_SIZE,
    ):
        self.entries = entries
        # Augmentation renforcée : multiplie virtuellement les ~30 cas positifs.
        # RandomAffine combine rotation + translation + zoom léger.
        aug = (
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.95, 1.05),
                ),
                transforms.ColorJitter(brightness=0.25, contrast=0.25),
            ]
            if augment
            else []
        )
        self.transform = transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((img_size, img_size)),   # resize anisotrope → pas de perte de tissu
                *aug,
                transforms.ToTensor(),
                transforms.Normalize(mean, std),           # stats propres au dataset mammographie
            ]
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        path, label = self.entries[idx]
        img = Image.open(path).convert("L")
        return self.transform(img), torch.tensor(label, dtype=torch.float32)


def compute_dataset_stats(
    entries: list[tuple[str, int]],
    img_size: int,
    cache_path: Path,
) -> tuple[list[float], list[float]]:
    """Calcule mean/std des images à la résolution cible, avec cache sur disque."""
    if cache_path.exists():
        stats = json.loads(cache_path.read_text())
        print(f"Stats chargées depuis {cache_path.name} : mean={stats['mean']}  std={stats['std']}")
        return stats["mean"], stats["std"]

    raw = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    ch_sum = torch.zeros(3)
    ch_sq  = torch.zeros(3)
    n_pix  = 0
    for path, _ in tqdm(entries, desc="Calcul mean/std train", unit="img"):
        t = raw(Image.open(path).convert("L"))              # (3, H, W) ∈ [0, 1]
        ch_sum += t.sum(dim=[1, 2])
        ch_sq  += (t ** 2).sum(dim=[1, 2])
        n_pix  += t.shape[1] * t.shape[2]
    mean = (ch_sum / n_pix).tolist()
    std  = torch.sqrt(ch_sq / n_pix - (ch_sum / n_pix) ** 2).tolist()

    cache_path.write_text(json.dumps({"mean": mean, "std": std}, indent=2))
    print(f"Stats train calculées : mean={mean}  std={std}  (sauvé → {cache_path.name})")
    return mean, std


def _make_sampler(entries: list) -> WeightedRandomSampler:
    labels = [e[1] for e in entries]
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    w_pos = 1.0 / n_pos if n_pos else 0.0
    w_neg = 1.0 / n_neg if n_neg else 0.0
    weights = [w_pos if l == 1 else w_neg for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ─── Modèle ──────────────────────────────────────────────────────────────────

def build_resnet18(device: str, pretrained: bool = False) -> nn.Module:
    """Construit ResNet18. `pretrained=True` charge les poids ImageNet (fine-tuning)."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)


# ─── Boucle d'entraînement ───────────────────────────────────────────────────

def train(
    epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    img_size: int = IMG_SIZE,
    device: str = DEVICE,
    patience: int = EARLY_STOP_PATIENCE,
    pretrained: bool = False,
) -> None:
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    # Dossier de run isolé : chaque entraînement a ses propres artefacts.
    run_dir   = _make_run_dir("pretrained" if pretrained else "scratch")
    ckpt_path = run_dir / "best.pt"
    logs_path = run_dir / "logs.json"
    roc_path  = run_dir / "roc.png"
    args_path = run_dir / "args.json"
    # Le cache des stats reste global (dépend uniquement de img_size + split).
    stats_path = CHECKPOINT_DIR / f"train_stats_{img_size}.json"

    # Split stratifié au niveau exam (même logique que le pipeline GMIC)
    train_exams, val_exams = load_and_split(EXAM_LIST_PATH)

    train_entries = _make_entries(train_exams)
    val_entries = _make_entries(val_exams)

    n_cancer_train = sum(e[1] for e in train_entries)
    n_cancer_val = sum(e[1] for e in val_entries)
    print(f"Run dir : {run_dir}")
    print(f"Train : {len(train_entries)} images  |  cancer={n_cancer_train}")
    print(f"Val   : {len(val_entries)} images  |  cancer={n_cancer_val}")
    print(f"Device: {device}  |  batch_size={batch_size}  |  img_size={img_size}x{img_size}")
    print(f"lr={lr}  |  weight_decay={weight_decay}  |  early-stop patience={patience}")
    print(f"pretrained={pretrained}  |  warmup_epochs={WARMUP_EPOCHS}")
    print()

    # Stats train (calculées une fois, mises en cache)
    mean, std = compute_dataset_stats(train_entries, img_size, stats_path)

    train_ds = ImageDataset(train_entries, mean=mean, std=std, augment=True,  img_size=img_size)
    val_ds   = ImageDataset(val_entries,   mean=mean, std=std, augment=False, img_size=img_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=_make_sampler(train_entries),
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = build_resnet18(device, pretrained=pretrained)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Warm-up linéaire (5 epochs) puis cosine annealing — évite l'oscillation initiale
    # quand on part d'une init aléatoire ou de poids pré-entraînés.
    warmup_iters = min(WARMUP_EPOCHS, max(1, epochs - 1))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_iters
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs - warmup_iters)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_iters]
    )
    loss_fn = nn.BCEWithLogitsLoss()

    hyperparams = {
        "epochs": epochs, "batch_size": batch_size, "lr": lr,
        "weight_decay": weight_decay, "img_size": img_size, "device": device,
        "pretrained": pretrained, "warmup_epochs": warmup_iters,
        "n_train": len(train_entries), "n_val": len(val_entries),
        "n_cancer_train": n_cancer_train, "n_cancer_val": n_cancer_val,
        "augmentation": "hflip + affine(rot=10, trans=5%, scale=±5%) + jitter(0.25)",
    }
    args_path.write_text(json.dumps(hyperparams, indent=2))
    logs = {"hyperparams": hyperparams, "epochs": []}

    best_auc = 0.0
    best_epoch = 0
    epochs_since_best = 0
    for epoch in range(1, epochs + 1):
        # — train
        model.train()
        train_loss = 0.0
        t0 = time.time()
        bar = tqdm(train_loader, desc=f"Epoch {epoch:3d}/{epochs} [train]",
                   unit="batch", leave=False)
        for imgs, labels in bar:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs).squeeze(1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")
        scheduler.step()
        train_time = time.time() - t0

        # — validation
        model.eval()
        val_loss = 0.0
        preds, targets = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch:3d}/{epochs} [val] ",
                                     unit="batch", leave=False):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs).squeeze(1)
                val_loss += loss_fn(logits, labels).item()
                preds.extend(torch.sigmoid(logits).cpu().tolist())
                targets.extend(labels.cpu().tolist())

        auc = (
            roc_auc_score(targets, preds)
            if len(set(targets)) > 1
            else float("nan")
        )
        is_best = not np.isnan(auc) and auc > best_auc
        if is_best:
            best_auc = auc
            best_epoch = epoch
            epochs_since_best = 0
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "val_auc": auc,
                    "val_preds": preds,
                    "val_targets": targets,
                },
                ckpt_path,
            )
        else:
            epochs_since_best += 1

        logs["epochs"].append({
            "epoch": epoch,
            "train_loss": round(train_loss / len(train_loader), 6),
            "val_loss": round(val_loss / len(val_loader), 6),
            "auc": round(auc, 6) if not np.isnan(auc) else None,
            "time_s": round(train_time),
            "is_best": is_best,
        })
        logs_path.write_text(json.dumps(logs, indent=2))

        flag = " ← best" if is_best else ""
        print(
            f"[{epoch:3d}/{epochs}]  "
            f"train_loss={train_loss / len(train_loader):.4f}  "
            f"val_loss={val_loss / len(val_loader):.4f}  "
            f"auc={auc:.4f}  "
            f"time={train_time:.0f}s{flag}"
        )

        if epochs_since_best >= patience:
            print(f"\nEarly stopping : pas d'amélioration depuis {patience} epochs "
                  f"(best={best_auc:.4f} à epoch {best_epoch}).")
            break

    print(f"\nMeilleur AUC val : {best_auc:.4f}  (epoch {best_epoch})  →  {ckpt_path}")

    # — courbe ROC du meilleur epoch (val set uniquement)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    best_preds   = ckpt["val_preds"]
    best_targets = ckpt["val_targets"]

    if len(set(best_targets)) < 2:
        print("Pas assez de classes distinctes pour tracer la courbe ROC.")
        return

    fpr, tpr, _ = roc_curve(best_targets, best_preds)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {best_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard (0.5)")
    ax.set_xlabel("Taux faux positifs")
    ax.set_ylabel("Taux vrais positifs")
    ax.set_title(f"Courbe ROC — val set (meilleur epoch : {ckpt['epoch']})")
    ax.legend()
    fig.savefig(roc_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Courbe ROC sauvegardée → {roc_path}")
    print(f"\nTous les artefacts du run : {run_dir}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraîne ResNet18 sur les mammographies RSNA.")
    parser.add_argument("--epochs",       type=int,   default=NUM_EPOCHS,    help=f"Nombre d'epochs (défaut: {NUM_EPOCHS})")
    parser.add_argument("--batch-size",   type=int,   default=BATCH_SIZE,    help=f"Taille de batch (défaut: {BATCH_SIZE})")
    parser.add_argument("--lr",           type=float, default=None,          help=f"Learning rate (défaut: {LEARNING_RATE} from-scratch, {LR_PRETRAINED} si --pretrained)")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,  help=f"L2 regularisation (défaut: {WEIGHT_DECAY})")
    parser.add_argument("--img-size",     type=int,   default=IMG_SIZE,      help=f"Taille des images carrées (défaut: {IMG_SIZE})")
    parser.add_argument("--device",       type=str,   default=DEVICE,        help=f"Device pytorch (défaut: {DEVICE})")
    parser.add_argument("--patience",     type=int,   default=EARLY_STOP_PATIENCE, help=f"Epochs sans amélioration avant early stop (défaut: {EARLY_STOP_PATIENCE})")
    parser.add_argument("--pretrained",   action="store_true",               help="Charge les poids ImageNet (fine-tuning) au lieu de from-scratch")
    args = parser.parse_args()

    # Si --lr non fourni : LR_PRETRAINED en fine-tuning, LEARNING_RATE sinon.
    lr = args.lr if args.lr is not None else (LR_PRETRAINED if args.pretrained else LEARNING_RATE)

    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=lr,
        weight_decay=args.weight_decay,
        img_size=args.img_size,
        device=args.device,
        patience=args.patience,
        pretrained=args.pretrained,
    )
