"""
Tests unitaires pour la validation des donnees d'entree.
Lance avec : make test  ou  python -m pytest tests/ -v
"""

import os
import csv
import tempfile
import shutil

import cv2
import numpy as np
import pytest

# Ajuster le path pour importer validate_input
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from validate_input import (
    ValidationResult,
    check_csv,
    check_image,
    check_views,
)


@pytest.fixture
def tmp_dir():
    """Cree un dossier temporaire pour les tests."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d)


@pytest.fixture
def valid_csv(tmp_dir):
    """Cree un CSV valide minimal."""
    path = os.path.join(tmp_dir, "train.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["patient_id", "image_id", "laterality", "view", "cancer"])
        w.writerow(["10001", "111", "L", "CC", 0])
        w.writerow(["10001", "112", "L", "MLO", 0])
        w.writerow(["10001", "113", "R", "CC", 0])
        w.writerow(["10001", "114", "R", "MLO", 0])
        w.writerow(["10002", "211", "L", "CC", 1])
        w.writerow(["10002", "212", "L", "MLO", 1])
        w.writerow(["10002", "213", "R", "CC", 1])
        w.writerow(["10002", "214", "R", "MLO", 1])
    return path


def make_grayscale_image(path, h=3000, w=2000, dtype=np.uint16):
    """Cree une image grayscale valide (simule une mammographie)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(100, 60000, (h, w), dtype=dtype)
    cv2.imwrite(path, img)
    return path


def make_rgb_image(path, h=3000, w=2000):
    """Cree une image couleur RGB (photo classique)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def make_tiny_image(path, h=100, w=80):
    """Cree une image trop petite."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.random.randint(0, 255, (h, w), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


def make_black_image(path, h=3000, w=2000):
    """Cree une image toute noire."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = np.zeros((h, w), dtype=np.uint8)
    cv2.imwrite(path, img)
    return path


# ── Tests CSV ────────────────────────────────────────────────────────────────

class TestCSV:
    def test_valid_csv(self, valid_csv):
        result = ValidationResult()
        df = check_csv(valid_csv, result)
        assert result.passed
        assert df is not None
        assert len(df) == 8

    def test_missing_csv(self):
        result = ValidationResult()
        df = check_csv("/nonexistent/train.csv", result)
        assert not result.passed
        assert df is None

    def test_csv_is_a_directory(self, tmp_dir):
        # Un dossier nomme train.csv (cas reel rencontre dans Downloads)
        fake_csv = os.path.join(tmp_dir, "train.csv")
        os.makedirs(fake_csv)
        result = ValidationResult()
        df = check_csv(fake_csv, result)
        assert not result.passed
        assert any("dossier" in e for e in result.errors)

    def test_missing_columns(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_id"])
            w.writerow(["10001", "111"])
        result = ValidationResult()
        df = check_csv(path, result)
        assert not result.passed
        assert "Colonnes manquantes" in result.errors[0]

    def test_invalid_laterality(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad_lat.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_id", "laterality", "view", "cancer"])
            w.writerow(["10001", "111", "Left", "CC", 0])
        result = ValidationResult()
        check_csv(path, result)
        assert not result.passed
        assert any("Lateralites invalides" in e for e in result.errors)

    def test_invalid_view(self, tmp_dir):
        path = os.path.join(tmp_dir, "bad_view.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_id", "laterality", "view", "cancer"])
            w.writerow(["10001", "111", "L", "XRAY", 0])
        result = ValidationResult()
        check_csv(path, result)
        assert not result.passed
        assert any("Vues invalides" in e for e in result.errors)


# ── Tests Images ─────────────────────────────────────────────────────────────

class TestImages:
    def test_valid_grayscale(self, tmp_dir):
        path = make_grayscale_image(os.path.join(tmp_dir, "img.png"))
        result = ValidationResult()
        assert check_image(path, result) is True
        assert result.passed

    def test_rgb_image_rejected(self, tmp_dir):
        path = make_rgb_image(os.path.join(tmp_dir, "color.png"))
        result = ValidationResult()
        assert check_image(path, result) is False
        assert not result.passed
        assert any("COULEUR" in e for e in result.errors)

    def test_tiny_image_rejected(self, tmp_dir):
        path = make_tiny_image(os.path.join(tmp_dir, "tiny.png"))
        result = ValidationResult()
        assert check_image(path, result) is False
        assert any("trop petite" in e for e in result.errors)

    def test_black_image_rejected(self, tmp_dir):
        path = make_black_image(os.path.join(tmp_dir, "black.png"))
        result = ValidationResult()
        assert check_image(path, result) is False
        assert any("uniforme" in e for e in result.errors)

    def test_corrupted_file(self, tmp_dir):
        path = os.path.join(tmp_dir, "corrupt.png")
        with open(path, "w") as f:
            f.write("this is not an image")
        result = ValidationResult()
        assert check_image(path, result) is False
        assert any("illisible" in e for e in result.errors)


# ── Tests Vues ───────────────────────────────────────────────────────────────

class TestViews:
    def test_complete_views(self, valid_csv):
        import pandas as pd
        df = pd.read_csv(valid_csv)
        result = ValidationResult()
        check_views(df, result)
        assert result.passed
        assert any("4 vues" in msg for msg in result.info)

    def test_missing_views_warns(self, tmp_dir):
        import pandas as pd
        path = os.path.join(tmp_dir, "incomplete.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_id", "laterality", "view", "cancer"])
            # Patient avec seulement le cote gauche
            w.writerow(["10001", "111", "L", "CC", 0])
            w.writerow(["10001", "112", "L", "MLO", 0])
        df = pd.read_csv(path)
        result = ValidationResult()
        check_views(df, result, strict=False)
        assert len(result.warnings) > 0
        assert any("4 vues" in w for w in result.warnings)

    def test_missing_views_strict_fails(self, tmp_dir):
        import pandas as pd
        path = os.path.join(tmp_dir, "incomplete.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["patient_id", "image_id", "laterality", "view", "cancer"])
            w.writerow(["10001", "111", "L", "CC", 0])
        df = pd.read_csv(path)
        result = ValidationResult()
        check_views(df, result, strict=True)
        assert not result.passed
