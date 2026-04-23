"""
Entraînement de la tête de fine-tuning sur les features GMIC cachées.

Principe :
  1. Charge (ou calcule) le cache de features GMIC (features.py)
  2. Split stratifié train / val (même seed que dataset.py → cohérent)
  3. Entraîne FineTuneHead avec BCEWithLogitsLoss
     - Échantillonnage 50/50 sur le train (WeightedRandomSampler)
     - Val : distribution réelle → AUC-ROC non biaisé
  4. Sauvegarde le meilleur modèle (meilleur AUC-ROC val)

Usage :
  python -m fine_tuning.train
"""

import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from fine_tuning.config import (
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    VAL_SPLIT,
    RANDOM_SEED,
    CHECKPOINT_DIR,
    DEVICE,
)
from fine_tuning.features import extract_and_cache
from fine_tuning.model import FineTuneHead
from fine_tuning.loss import make_loss


BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "head_best.pt")


def _split_indices(labels: torch.Tensor):
    """Split stratifié (ou aléatoire si trop peu de positifs)."""
    n = len(labels)
    y = labels.numpy().astype(int)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    can_stratify = n_pos >= 2 and n_neg >= 2

    idx_train, idx_val = train_test_split(
        np.arange(n),
        test_size=VAL_SPLIT,
        stratify=y if can_stratify else None,
        random_state=RANDOM_SEED,
    )
    return idx_train, idx_val


def _make_balanced_sampler(labels: torch.Tensor) -> WeightedRandomSampler:
    y = labels.numpy().astype(int)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    w_pos = 1.0 / n_pos if n_pos > 0 else 0.0
    w_neg = 1.0 / n_neg if n_neg > 0 else 0.0
    weights = [w_pos if lbl == 1 else w_neg for lbl in y]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: str):
    """Retourne (loss_moy, auc_roc, accuracy) sur le loader."""
    model.eval()
    loss_fn = make_loss()
    all_logits, all_labels, total_loss = [], [], 0.0
    n = 0
    for feats, lbls in loader:
        feats = feats.to(device)
        lbls = lbls.to(device)
        logits = model(feats)
        total_loss += loss_fn(logits, lbls).item() * feats.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(lbls.cpu())
        n += feats.size(0)

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy().astype(int)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs > 0.5).astype(int)

    auc = roc_auc_score(y_true, probs) if len(set(y_true.tolist())) == 2 else float("nan")
    acc = float((preds == y_true).mean())
    return total_loss / n, auc, acc


def train():
    warnings.filterwarnings("ignore", category=UserWarning)

    device = DEVICE if torch.cuda.is_available() else "cpu"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 1. Cache des features GMIC (calculé si absent)
    cache = extract_and_cache(device=device)
    features, labels = cache["features"], cache["labels"]

    # 2. Split train / val
    idx_train, idx_val = _split_indices(labels)
    X_train, y_train = features[idx_train], labels[idx_train]
    X_val,   y_val   = features[idx_val],   labels[idx_val]

    n_pos_tr, n_pos_va = int(y_train.sum()), int(y_val.sum())
    print(f"\nTrain : {len(y_train)} exams  |  cancer={n_pos_tr}  sain={len(y_train)-n_pos_tr}")
    print(f"Val   : {len(y_val)} exams  |  cancer={n_pos_va}  sain={len(y_val)-n_pos_va}")

    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)
    sampler  = _make_balanced_sampler(y_train)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader   = DataLoader(val_ds,   batch_size=max(BATCH_SIZE, 8), shuffle=False)

    # 3. Modèle + optim
    torch.manual_seed(RANDOM_SEED)
    model = FineTuneHead().to(device)
    loss_fn = make_loss()
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print(f"\nModèle : {sum(p.numel() for p in model.parameters())} paramètres entraînables")
    print(f"Device : {device}")
    print(f"Epochs : {NUM_EPOCHS}  |  lr={LEARNING_RATE}  |  wd={WEIGHT_DECAY}\n")

    # 4. Boucle d'entraînement
    best_auc = -1.0
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        total, n = 0.0, 0
        for feats, lbls in train_loader:
            feats = feats.to(device)
            lbls = lbls.to(device)
            optim.zero_grad()
            logits = model(feats)
            loss = loss_fn(logits, lbls)
            loss.backward()
            optim.step()
            total += loss.item() * feats.size(0)
            n += feats.size(0)
        tr_loss = total / n

        val_loss, val_auc, val_acc = _evaluate(model, val_loader, device)
        flag = ""
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            torch.save({"state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_auc": val_auc}, BEST_MODEL_PATH)
            flag = "  <-- best"

        auc_str = f"{val_auc:.4f}" if not np.isnan(val_auc) else "n/a"
        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
              f"val_auc={auc_str}  val_acc={val_acc:.3f}{flag}")

    print(f"\nBest val AUC-ROC : {best_auc:.4f}")
    print(f"Checkpoint : {BEST_MODEL_PATH}")


if __name__ == "__main__":
    train()
