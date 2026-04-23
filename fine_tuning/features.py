"""
Extraction et cache des features GMIC.

Idée : puisque GMIC est gelé, ses sorties sur une image donnée ne changent
jamais. On les calcule UNE SEULE FOIS et on les sauvegarde sur disque.
→ l'entraînement devient quasi-instantané (on n'entraîne qu'une petite MLP).

Produit un fichier features.pt contenant :
  {
    "features": Tensor (N, 8)  — [benign, mal] pour chaque des 4 vues
    "labels":   Tensor (N,)    — 0/1 malignant exam-level
    "image_size": (H, W)
  }

Usage :
  python -m fine_tuning.features
"""

import os
import argparse
import warnings

import torch
from tqdm import tqdm

from fine_tuning.config import (
    EXAM_LIST_PATH,
    IMAGE_DIR,
    IMAGE_SIZE,
    CHECKPOINT_DIR,
    DEVICE,
    PROJECT_ROOT,
    RUN_NAME,
)
from fine_tuning.dataset import MammographyExamDataset
from fine_tuning.model import load_gmic_backbone


GMIC_DIR = os.path.join(PROJECT_ROOT, "GMIC")
CACHE_PATH = os.path.join(CHECKPOINT_DIR, "features.pt")


def _extract_exam_features(gmic, images: torch.Tensor, device: str) -> torch.Tensor:
    """
    images : Tensor (4, H, W) — un exam, déjà aligné et normalisé par le dataset
    returns: Tensor (8,) — concat des sorties GMIC des 4 vues
    """
    feats = []
    with torch.no_grad():
        for v in range(4):
            x = images[v].unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
            out = gmic(x).cpu().squeeze(0)                      # (2,)
            feats.append(out)
    return torch.cat(feats, dim=0)                              # (8,)


def extract_and_cache(device: str = None, force: bool = False, max_exams: int = None) -> dict:
    """
    Calcule les features GMIC pour TOUS les exams.
    Le split train/val est fait au moment de l'entraînement à partir du cache.
    """
    if device is None:
        device = DEVICE if torch.cuda.is_available() else "cpu"

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    if os.path.exists(CACHE_PATH) and not force:
        print(f"Cache trouvé : {CACHE_PATH}")
        cache = torch.load(CACHE_PATH, weights_only=False)

        cache_image_size = tuple(cache.get("image_size", ()))
        cache_exam_list_path = cache.get("exam_list_path")
        cache_run_name = cache.get("run_name")
        cache_data_mtime = float(cache.get("data_mtime", -1.0))

        current_data_mtime = os.path.getmtime(EXAM_LIST_PATH) if os.path.exists(EXAM_LIST_PATH) else -1.0

        same_image_size = cache_image_size == tuple(IMAGE_SIZE)
        same_exam_list = cache_exam_list_path == EXAM_LIST_PATH
        same_run = cache_run_name == RUN_NAME
        data_not_newer = current_data_mtime <= cache_data_mtime

        if same_image_size and same_exam_list and same_run and data_not_newer:
            print(f"  {cache['features'].shape[0]} exams déjà cachés — on réutilise.")
            return cache

        print("  Cache invalide pour la config courante -> recalcul.")
        if not same_image_size:
            print(f"    - IMAGE_SIZE cache={cache_image_size} != config={tuple(IMAGE_SIZE)}")
        if not same_exam_list:
            print(f"    - EXAM_LIST_PATH cache={cache_exam_list_path} != config={EXAM_LIST_PATH}")
        if not same_run:
            print(f"    - RUN_NAME cache={cache_run_name} != config={RUN_NAME}")
        if not data_not_newer:
            print("    - data.pkl plus récent que le cache")

    print(f"Extraction des features GMIC (device={device})")
    print(f"  IMAGE_SIZE = {IMAGE_SIZE}")

    # Charge TOUS les exams dans leur ordre original (pkl).
    # Le split train/val est fait plus tard, dans train.py, à partir du cache.
    import pickle
    with open(EXAM_LIST_PATH, "rb") as f:
        all_exams = pickle.load(f)

    if max_exams is not None:
        all_exams = all_exams[:max_exams]
        print(f"  [mode test] limité à {len(all_exams)} exams")

    dataset = MammographyExamDataset(
        exam_list=all_exams,
        image_dir=IMAGE_DIR,
        image_size=IMAGE_SIZE,
        augment=False,
    )

    gmic = load_gmic_backbone(GMIC_DIR, IMAGE_SIZE, device=device)

    features = torch.zeros(len(dataset), 8, dtype=torch.float32)
    labels = torch.zeros(len(dataset), dtype=torch.float32)

    for i in tqdm(range(len(dataset)), desc="GMIC forward"):
        imgs, lbl = dataset[i]
        features[i] = _extract_exam_features(gmic, imgs, device)
        labels[i] = lbl

    cache = {
        "features": features,
        "labels": labels,
        "image_size": tuple(IMAGE_SIZE),
        "run_name": RUN_NAME,
        "exam_list_path": EXAM_LIST_PATH,
        "data_mtime": os.path.getmtime(EXAM_LIST_PATH) if os.path.exists(EXAM_LIST_PATH) else -1.0,
    }
    if max_exams is None:
        torch.save(cache, CACHE_PATH)
        print(f"\nFeatures sauvegardées : {CACHE_PATH}")
    else:
        print(f"\n[mode test] cache NON sauvegardé (max_exams={max_exams})")
    print(f"  shape features = {features.shape}  |  cancer={int(labels.sum())} / sain={int((1-labels).sum())}")
    return cache


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    parser = argparse.ArgumentParser(description="Extraction et cache des features GMIC")
    parser.add_argument("--force", action="store_true", help="Forcer le recalcul du cache")
    parser.add_argument("--max-exams", type=int, default=None,
                        help="Limiter le nombre d'exams (debug) sans sauvegarder le cache")
    parser.add_argument("--device", type=str, default=None,
                        help="Device a utiliser (cuda ou cpu). Defaut: auto")
    args = parser.parse_args()

    extract_and_cache(device=args.device, force=args.force, max_exams=args.max_exams)
