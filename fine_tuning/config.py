"""
Configuration centrale du fine-tuning.
Modifier ce fichier pour changer les chemins et hyperparamètres.
"""

import os

# ─── Chemins ────────────────────────────────────────────────────────────────

# Dossier racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Pkl contenant les exams preprocessés (cropped_exam_list.pkl)
EXAM_LIST_PATH = os.path.join(PROJECT_ROOT, "data", "extract_dataset", "cropped_exam_list.pkl")

# Dossier contenant les images PNG : {IMAGE_DIR}/{patient_id}/{image_id}.png
IMAGE_DIR = os.path.join(PROJECT_ROOT, "data", "extract_dataset", "png_images")

# Dossier où sauvegarder les checkpoints du modèle
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "fine_tuning", "checkpoints")

# ─── Split train / validation ────────────────────────────────────────────────

# Proportion des exams utilisée pour la validation (le reste = train)
VAL_SPLIT = 0.2

# Graine aléatoire pour reproductibilité du split
RANDOM_SEED = 42

# ─── Images ─────────────────────────────────────────────────────────────────

# Taille de redimensionnement (H, W) des images en entrée du modèle.
# Les originales font 2944x1920 — trop grand pour 4GB VRAM.
# Réduire ici si OOM (Out Of Memory).
IMAGE_SIZE = (1472, 960)   # moitié de la résolution originale

# ─── Entraînement ────────────────────────────────────────────────────────────

# Taille de batch (1 exam = 4 images). Avec 4GB VRAM, garder à 1 ou 2.
BATCH_SIZE = 1

# Nombre d'epochs
NUM_EPOCHS = 20

# Taux d'apprentissage initial
LEARNING_RATE = 1e-4

# Poids L2 (régularisation)
WEIGHT_DECAY = 1e-5

# Nombre de workers pour le DataLoader (0 = pas de multiprocessing)
NUM_WORKERS = 2

# Utiliser le GPU si disponible
DEVICE = "cuda"   # "cpu" pour forcer le CPU
