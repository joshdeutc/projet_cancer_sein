"""
Fonction de perte.

Choix : BCEWithLogitsLoss (Binary Cross-Entropy avec logits).

Pourquoi pas Focal Loss ni pos_weight ?
  - Notre WeightedRandomSampler (dataset.py) tire déjà 50% cancer / 50% sain
    dans chaque batch → le déséquilibre est géré en amont.
  - Ajouter pos_weight ou Focal Loss par-dessus revient à sur-pondérer
    deux fois la classe minoritaire → instabilité d'entraînement.
  - BCEWithLogitsLoss = numériquement stable (logit → sigmoid → BCE fusionnés).

La tête du modèle doit donc renvoyer des logits (pas de sigmoid final).
"""

import torch.nn as nn


def make_loss():
    """Renvoie la fonction de perte utilisée pour l'entraînement."""
    return nn.BCEWithLogitsLoss()
