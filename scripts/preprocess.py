"""
Prétraitement GMIC — Étapes 1 à 4
-----------------------------------
Prépare les images (DICOM ou PNG) pour l'inférence GMIC :

  1. Détection du format d'entrée (DICOM ou PNG)
  2. Conversion DICOM -> PNG 16-bit (si nécessaire)
  3. Construction du PKL au format GMIC
  4. Recadrage (crop_mammogram.py)
  5. Redimensionnement à 2944×1920 + normalisation uint8

Le flip horizontal (vues droites) et la normalisation mean/std sont appliqués
directement à l'inférence — inutile de les précalculer.

Le dossier de sortie contiendra :
  <output-dir>/cropped_images/         <- images prêtes pour l'inférence
  <output-dir>/data.pkl                <- PKL final (à passer à inference.py)
  <output-dir>/exam_list_before_cropping.pkl
  <output-dir>/cropped_exam_list.pkl

Usage :
  python scripts/preprocess.py --input-dir data/demo --output-dir output/demo
  python scripts/preprocess.py --input-dir data/extract_dataset --output-dir output/dicom
  python scripts/preprocess.py --input-dir data/demo --output-dir output/demo --force-crop
"""

import os
import sys
import csv
import glob
import pickle
import shutil
import random
import zipfile
import argparse

import cv2
import numpy as np
import pydicom
from tqdm import tqdm

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GMIC_DIR = os.path.join(PROJECT_DIR, "GMIC")

GMIC_H, GMIC_W = 2944, 1920

sys.path.insert(0, GMIC_DIR)


# ── Détection automatique du format ──────────────────────────────────────────

def detect_format(input_dir: str) -> str:
    search_dir = input_dir
    train_images = os.path.join(input_dir, "train_images")
    if os.path.isdir(train_images):
        search_dir = train_images

    for patient_dir in sorted(os.listdir(search_dir))[:20]:
        patient_path = os.path.join(search_dir, patient_dir)
        if not os.path.isdir(patient_path):
            continue
        for f in os.listdir(patient_path):
            if f.endswith(".dcm.zip") or f.endswith(".dcm"):
                return "dicom"
            if f.endswith(".png"):
                return "png"

    return "unknown"


# ── Étape 1 : Conversion DICOM -> PNG 16-bit ─────────────────────────────────

def convert_dcm_to_png(raw_dir: str, png_dir: str, csv_path: str):
    print("\n" + "=" * 60)
    print("ETAPE 1 : Conversion DICOM -> PNG 16-bit")
    print("=" * 60)

    image_ids = set()
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            image_ids.add((row["patient_id"], row["image_id"]))

    converted, skipped, failed = 0, 0, 0

    for pid, iid in tqdm(sorted(image_ids), desc="Conversion"):
        out_path = os.path.join(png_dir, pid, f"{iid}.png")
        if os.path.exists(out_path):
            skipped += 1
            continue

        zip_path = os.path.join(raw_dir, pid, f"{iid}.dcm.zip")
        dcm_path = os.path.join(raw_dir, pid, f"{iid}.dcm")

        os.makedirs(os.path.join(png_dir, pid), exist_ok=True)

        try:
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, "r") as z:
                    dcm_name = next(n for n in z.namelist() if n.endswith(".dcm"))
                    with z.open(dcm_name) as f_dcm:
                        ds = pydicom.dcmread(f_dcm)
            elif os.path.exists(dcm_path):
                ds = pydicom.dcmread(dcm_path)
            else:
                failed += 1
                continue

            arr = ds.pixel_array.astype(np.float32)
            if ds.PhotometricInterpretation == "MONOCHROME1":
                arr = arr.max() - arr
            arr_max = arr.max()
            if arr_max > 0:
                arr = (arr / arr_max * 65535).astype(np.uint16)
            else:
                arr = arr.astype(np.uint16)

            cv2.imwrite(out_path, arr)
            converted += 1

        except Exception as e:
            print(f"  Erreur {pid}/{iid} : {e}")
            failed += 1

    print(f"\nConvertis : {converted} | Deja OK : {skipped} | Echecs : {failed}")
    return converted + skipped


# ── Étape 2 : Construction du PKL GMIC ───────────────────────────────────────

def build_exam_pkl(csv_path: str, png_dir: str, pkl_path: str) -> list:
    print("\n" + "=" * 60)
    print("ETAPE 2 : Construction du PKL GMIC")
    print("=" * 60)

    patients: dict = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            pid = row["patient_id"]
            if pid not in patients:
                patients[pid] = {
                    "horizontal_flip": "NO",
                    "L-CC": [], "L-MLO": [], "R-CC": [], "R-MLO": [],
                    "L-CC_benign_seg": [], "L-CC_malignant_seg": [],
                    "L-MLO_benign_seg": [], "L-MLO_malignant_seg": [],
                    "R-CC_benign_seg": [], "R-CC_malignant_seg": [],
                    "R-MLO_benign_seg": [], "R-MLO_malignant_seg": [],
                    "cancer_label": {
                        "benign": 0, "left_benign": 0, "right_benign": 0,
                        "malignant": 0, "left_malignant": 0, "right_malignant": 0,
                        "unknown": 0,
                    },
                }

            iid = row["image_id"]
            lat = row["laterality"]
            view = row["view"]
            cancer = int(row["cancer"])
            view_key = f"{lat}-{view}"

            png_path = os.path.join(png_dir, pid, f"{iid}.png")
            if not os.path.exists(png_path):
                continue

            if view_key in patients[pid]:
                patients[pid][view_key].append(f"{pid}/{iid}")

            side = "left" if lat == "L" else "right"
            if cancer == 1:
                patients[pid]["cancer_label"]["malignant"] = 1
                patients[pid]["cancer_label"][f"{side}_malignant"] = 1
            else:
                patients[pid]["cancer_label"]["benign"] = 1
                patients[pid]["cancer_label"][f"{side}_benign"] = 1

    exam_list = list(patients.values())

    with open(pkl_path, "wb") as f:
        pickle.dump(exam_list, f)

    cancer_count = sum(1 for e in exam_list if e["cancer_label"]["malignant"])
    print(f"Examens : {len(exam_list)} total | {cancer_count} cancer | {len(exam_list)-cancer_count} sains")
    print(f"PKL sauvegarde : {pkl_path}")
    return exam_list


# ── Étape 3 : Crop ───────────────────────────────────────────────────────────

def run_crop(png_dir: str, cropped_dir: str, pkl_raw: str, pkl_cropped: str):
    print("\n" + "=" * 60)
    print("ETAPE 3 : Recadrage (crop_mammogram)")
    print("=" * 60)

    if os.path.exists(cropped_dir):
        print(f"Suppression de l'ancien dossier {cropped_dir}...")
        shutil.rmtree(cropped_dir)

    with open(pkl_raw, "rb") as f:
        exam_list_raw = pickle.load(f)
    total_images = sum(
        len(exam[v]) for exam in exam_list_raw
        for v in ["L-CC", "R-CC", "L-MLO", "R-MLO"]
    )
    print(f"Images a cropper : {total_images}")

    cmd = (
        f"cd {GMIC_DIR} && PYTHONPATH={GMIC_DIR}:$PYTHONPATH {sys.executable} "
        f"src/cropping/crop_mammogram.py "
        f"--input-data-folder {png_dir} "
        f"--output-data-folder {cropped_dir} "
        f"--exam-list-path {pkl_raw} "
        f"--cropped-exam-list-path {pkl_cropped} "
        f"--num-processes 4"
    )

    import subprocess
    import threading
    import time as _time

    proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    def _progress():
        while proc.poll() is None:
            done = len(glob.glob(os.path.join(cropped_dir, "**", "*.png"), recursive=True))
            print(f"\r  Croppes : {done}/{total_images}", end="", flush=True)
            _time.sleep(1)
        done = len(glob.glob(os.path.join(cropped_dir, "**", "*.png"), recursive=True))
        print(f"\r  Croppes : {done}/{total_images}", flush=True)

    t = threading.Thread(target=_progress, daemon=True)
    t.start()
    proc.wait()
    t.join()

    if proc.returncode == 0:
        print("Crop termine")
    else:
        err = proc.stderr.read().decode()
        print(f"Crop echoue (code {proc.returncode})")
        if err:
            print(err[-500:])
        sys.exit(1)


# ── Étape 4 : Resize 2944×1920 + normalisation ───────────────────────────────

def resize_all(cropped_dir: str, pkl_cropped: str):
    print("\n" + "=" * 60)
    print(f"ETAPE 4 : Redimensionnement a {GMIC_H}x{GMIC_W} + normalisation [0, 255]")
    print("=" * 60)

    import src.utilities.pickling as pickling
    import src.utilities.data_handling as data_handling

    exam_list = pickling.unpickle_from_file(pkl_cropped)
    image_list = data_handling.unpack_exam_into_images(exam_list, cropped=True)

    scale_map = {}
    resized, skipped, normalized = 0, 0, 0

    for datum in tqdm(image_list, desc="Resize"):
        path = os.path.join(cropped_dir, datum["short_file_path"] + ".png")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]

        if h == GMIC_H and w == GMIC_W:
            if img.max() > 255 or img.dtype != np.uint8:
                img = _normalize_uint8(img)
                cv2.imwrite(path, img)
                normalized += 1
            skipped += 1
            continue

        scale_map[datum["short_file_path"]] = (GMIC_H / h, GMIC_W / w)
        interp = cv2.INTER_AREA if (h > GMIC_H or w > GMIC_W) else cv2.INTER_LINEAR
        img_resized = cv2.resize(img, (GMIC_W, GMIC_H), interpolation=interp)

        if img_resized.max() > 255 or img_resized.dtype != np.uint8:
            img_resized = _normalize_uint8(img_resized)
            normalized += 1

        cv2.imwrite(path, img_resized)
        resized += 1

    print(f"Redimensionnees : {resized} | Deja OK : {skipped} | Normalisees [0,255] : {normalized}")

    VIEWS_LIST = ["L-CC", "R-CC", "L-MLO", "R-MLO"]
    updated = 0
    for exam in exam_list:
        for view in VIEWS_LIST:
            if view not in exam or not exam[view]:
                continue
            for j in range(len(exam[view])):
                sfp = exam[view][j]
                rp = exam["rightmost_points"][view][j]
                bp = exam["bottommost_points"][view][j]
                wl = exam["window_location"][view][j]

                needs_fix = False
                if int(rp[1]) > GMIC_W:
                    needs_fix = True
                if int(bp[0]) > GMIC_H:
                    needs_fix = True
                if sfp in scale_map:
                    needs_fix = True

                if not needs_fix:
                    continue

                if sfp in scale_map:
                    sh, sw = scale_map[sfp]
                else:
                    # rightmost_points[1] ≈ largeur crop, bottommost_points[0] ≈ hauteur crop
                    orig_h = float(bp[0]) if bp[0] > 0 else GMIC_H
                    orig_w = float(rp[1]) if rp[1] > 0 else GMIC_W
                    sh = GMIC_H / orig_h
                    sw = GMIC_W / orig_w

                (ry1, ry2), rx = rp
                exam["rightmost_points"][view][j] = (
                    (int(round(float(ry1) * sh)), int(round(float(ry2) * sh))),
                    int(round(float(rx) * sw)),
                )

                by, (bx1, bx2) = bp
                exam["bottommost_points"][view][j] = (
                    int(round(float(by) * sh)),
                    (int(round(float(bx1) * sw)), int(round(float(bx2) * sw))),
                )

                # window_location = (y_top, y_bottom, x_left, x_right) dans l'image
                # originale pre-crop — ces coordonnees ne changent pas avec le resize.
                # On ne modifie pas window_location ici.

                updated += 1

    if updated > 0:
        pickling.pickle_to_file(pkl_cropped, exam_list)
        print(f"PKL mis a jour : {updated} images avec coordonnees recalculees")


def _normalize_uint8(img):
    img_f = img.astype(np.float32)
    img_min, img_max = img_f.min(), img_f.max()
    if img_max > img_min:
        return ((img_f - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    return np.zeros_like(img_f, dtype=np.uint8)


# ── Étape 5 : get_optimal_centers ────────────────────────────────────────────

def run_optimal_centers(cropped_dir: str, pkl_cropped: str, pkl_final: str):
    print("\n" + "=" * 60)
    print("ETAPE 5 : Calcul des centres optimaux")
    print("=" * 60)

    cmd = (
        f"cd {GMIC_DIR} && PYTHONPATH={GMIC_DIR}:$PYTHONPATH {sys.executable} "
        f"src/optimal_centers/get_optimal_centers.py "
        f"--cropped-exam-list-path {pkl_cropped} "
        f"--data-prefix {cropped_dir} "
        f"--output-exam-list-path {pkl_final} "
        f"--num-processes 4"
    )
    ret = os.system(cmd)
    if ret == 0:
        print("get_optimal_centers termine")
    else:
        print(f"get_optimal_centers echoue (code {ret})")
        sys.exit(1)


# ── Détection automatique des étapes déjà effectuées ─────────────────────────

def _count_pngs(directory: str) -> int:
    return len(glob.glob(os.path.join(directory, "**", "*.png"), recursive=True))


def is_crop_done(cropped_dir: str, pkl_cropped: str) -> bool:
    if not os.path.exists(pkl_cropped) or not os.path.exists(cropped_dir):
        return False
    return _count_pngs(cropped_dir) > 0


def is_resize_done(cropped_dir: str, pkl_cropped: str) -> bool:
    if not os.path.exists(cropped_dir) or not os.path.exists(pkl_cropped):
        return False
    pngs = glob.glob(os.path.join(cropped_dir, "**", "*.png"), recursive=True)
    if not pngs:
        return False
    sample = random.sample(pngs, min(20, len(pngs)))
    for p in sample:
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h != GMIC_H or w != GMIC_W:
            return False
        if img.max() > 255 or img.dtype != np.uint8:
            return False
    with open(pkl_cropped, "rb") as f:
        exams = pickle.load(f)
    import src.utilities.data_handling as dh
    images = dh.unpack_exam_into_images(exams, cropped=True)
    for d in random.sample(images, min(20, len(images))):
        rp = d.get("rightmost_points")
        if rp is not None and int(rp[1]) > GMIC_W:
            return False
        bp = d.get("bottommost_points")
        if bp is not None and int(bp[0]) > GMIC_H:
            return False
    return True


def copy_pkl_as_final(pkl_cropped: str, pkl_final: str):
    """Copie cropped_exam_list.pkl en data.pkl — aucun best_center nécessaire."""
    shutil.copy(pkl_cropped, pkl_final)
    print(f"data.pkl genere depuis cropped_exam_list.pkl")


def is_final_done(pkl_final: str) -> bool:
    return os.path.exists(pkl_final)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Prétraitement GMIC — Prépare les images pour l'inférence",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python scripts/preprocess.py --input-dir data/demo --output-dir output/demo
  python scripts/preprocess.py --input-dir data/extract_dataset --output-dir output/dicom
  python scripts/preprocess.py --input-dir data/demo --output-dir output/demo --force-crop
        """,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Dossier contenant les images (DICOM ou PNG)")
    parser.add_argument("--csv", default=None,
                        help="Chemin vers le CSV. Par defaut : <input-dir>/train_subset_test.csv ou train.csv")
    parser.add_argument("--output-dir", default=None,
                        help="Dossier de sortie. Par defaut : <projet>/output/")
    parser.add_argument("--format", choices=["dicom", "png", "auto"], default="auto",
                        help="Format des images d'entree (defaut: auto-detection)")
    parser.add_argument("--force-crop", action="store_true",
                        help="Forcer le crop meme s'il semble deja fait")
    parser.add_argument("--force-resize", action="store_true",
                        help="Forcer le resize meme s'il semble deja fait")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Erreur : dossier introuvable : {input_dir}")
        sys.exit(1)

    csv_path = args.csv
    if csv_path is None:
        csv_path = os.path.join(input_dir, "train_subset_test.csv")
        if not os.path.exists(csv_path):
            csv_path = os.path.join(input_dir, "train.csv")
    if not os.path.exists(csv_path):
        print(f"Erreur : CSV introuvable : {csv_path}")
        sys.exit(1)

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, "output")
    output_dir = os.path.abspath(output_dir)

    png_dir = os.path.join(output_dir, "png_images")
    cropped_dir = os.path.join(output_dir, "cropped_images")
    pkl_raw = os.path.join(output_dir, "exam_list_before_cropping.pkl")
    pkl_cropped = os.path.join(output_dir, "cropped_exam_list.pkl")
    pkl_final = os.path.join(output_dir, "data.pkl")

    os.makedirs(output_dir, exist_ok=True)

    fmt = args.format
    if fmt == "auto":
        fmt = detect_format(input_dir)
        if fmt == "unknown":
            print("Erreur : impossible de detecter le format des images.")
            print("Specifiez --format dicom ou --format png")
            sys.exit(1)
    print(f"Format detecte : {fmt.upper()}")

    if fmt == "dicom":
        raw_dir = os.path.join(input_dir, "train_images")
        if not os.path.isdir(raw_dir):
            raw_dir = input_dir
        os.makedirs(png_dir, exist_ok=True)
        convert_dcm_to_png(raw_dir, png_dir, csv_path)
        source_png_dir = png_dir
    else:
        train_images = os.path.join(input_dir, "train_images")
        source_png_dir = train_images if os.path.isdir(train_images) else input_dir

    # Sauvegarder le dossier source pour que le notebook puisse retrouver les images originales
    with open(os.path.join(output_dir, "source_dir.txt"), "w") as _f:
        _f.write(source_png_dir)

    build_exam_pkl(csv_path, source_png_dir, pkl_raw)

    if args.force_crop:
        run_crop(source_png_dir, cropped_dir, pkl_raw, pkl_cropped)
    elif is_crop_done(cropped_dir, pkl_cropped):
        n = _count_pngs(cropped_dir)
        print(f"\n[AUTO] Crop : {n} images deja croppees -> SKIP")
    else:
        run_crop(source_png_dir, cropped_dir, pkl_raw, pkl_cropped)

    if args.force_resize:
        resize_all(cropped_dir, pkl_cropped)
    elif is_resize_done(cropped_dir, pkl_cropped):
        print(f"[AUTO] Resize : images deja en {GMIC_H}x{GMIC_W} uint8 -> SKIP")
    else:
        resize_all(cropped_dir, pkl_cropped)

    if is_final_done(pkl_final):
        print(f"[AUTO] PKL final : {pkl_final} deja present -> SKIP")
    else:
        copy_pkl_as_final(pkl_cropped, pkl_final)

    print("\n" + "=" * 60)
    print("PREPROCESSING TERMINE")
    print(f"  Images cropppees : {_count_pngs(cropped_dir)}")
    print(f"  PKL final        : {pkl_final}")
    print(f"\nLancer l'inference avec :")
    print(f"  python scripts/inference.py --output-dir {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
