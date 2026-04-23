"""
Modèle de fine-tuning : GMIC gelé + petite tête entraînable.

Pourquoi ce design ?
  - Dataset très petit (119 exams, 5 cancers seulement)
  - Full fine-tuning = overfit garanti (millions de params vs 5 positifs)
  - Feature extraction : on garde GMIC intact, on entraîne uniquement
    un petit classifieur qui agrège les prédictions des 4 vues.

Flux :
  4 vues (H, W) → GMIC gelé → 4 × 2 probas (benign, malignant)
                            → concat 8-dim → MLP → 1 logit malignant
"""

import os
import sys

import torch
import torch.nn as nn


def load_gmic_backbone(gmic_dir: str, image_size: tuple, device: str = "cpu",
                       model_idx: str = "1"):
    """
    Charge GMIC pré-entraîné (sample_model_{idx}.p) avec cam_size adapté
    à IMAGE_SIZE. Le modèle est mis en eval() et gelé (requires_grad=False).

    Args:
        gmic_dir   : chemin vers le dossier GMIC (contient models/ et src/)
        image_size : (H, W) — doit être divisible par 64
        device     : "cpu" ou "cuda" — GMIC utilise ce flag en interne
                     (création de masques, crops) via params["device_type"]
        model_idx  : "1" à "5" — lequel des 5 modèles d'ensemble utiliser

    Returns:
        model : GMIC sur le device demandé, en mode eval, tous les poids gelés
    """
    if gmic_dir not in sys.path:
        sys.path.insert(0, gmic_dir)

    from src.modeling import gmic as gmic_module
    from src.constants import PERCENT_T_DICT

    H, W = image_size
    assert H % 64 == 0 and W % 64 == 0, \
        f"IMAGE_SIZE doit être divisible par 64, reçu {image_size}"

    on_gpu = device.startswith("cuda")

    params = {
        "device_type": "gpu" if on_gpu else "cpu",
        "gpu_number": 0,
        "cam_size": (H // 64, W // 64),
        "K": 6,
        "crop_shape": (256, 256),
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "percent_t": PERCENT_T_DICT[model_idx],
    }

    model = gmic_module.GMIC(params)
    weights_path = os.path.join(gmic_dir, "models", f"sample_model_{model_idx}.p")
    model.load_state_dict(
        torch.load(weights_path, map_location="cpu", weights_only=False),
        strict=False,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if on_gpu:
        model = model.cuda()
    return model


class FineTuneHead(nn.Module):
    """
    Petite tête MLP qui agrège les sorties GMIC des 4 vues en 1 logit.

    Input  : features (B, 8)  — [benign_LCC, mal_LCC, benign_LMLO, ..., mal_RMLO]
    Output : logits   (B,)    — logit de la classe "malignant" (exam-level)

    Architecture délibérément minimale (~150 params) pour éviter l'overfit
    sur 5 cas positifs.
    """

    def __init__(self, in_dim: int = 8, hidden_dim: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
