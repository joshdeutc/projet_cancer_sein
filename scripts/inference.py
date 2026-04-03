"""
Inférence GMIC — Ensemble de 5 modèles
----------------------------------------
Lance l'inférence sur les images prétraitées par preprocess.py et produit
les prédictions de cancer du sein.

Prérequis : avoir exécuté preprocess.py sur le même --output-dir.
Les fichiers attendus dans --output-dir :
  cropped_images/   <- images 2944×1920 prêtes
  data.pkl          <- PKL avec best_center

Résultats produits dans --output-dir :
  predictions.csv   <- scores malignant/benign + labels

Usage :
  python scripts/inference.py --output-dir output/demo
  python scripts/inference.py --output-dir output/demo --predictions-csv output/demo/results.csv
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
GMIC_DIR = os.path.join(PROJECT_DIR, "GMIC")

sys.path.insert(0, GMIC_DIR)


# ── Inférence ensemble 5 modèles ─────────────────────────────────────────────

def run_inference(cropped_dir: str, pkl_final: str, output_dir: str,
                  predictions_csv: str):
    print("\n" + "=" * 60)
    print("INFERENCE GMIC (ensemble 5 modeles)")
    print("=" * 60)

    import torch
    from src.modeling import gmic as gmic_module
    from src.scripts.run_model import run_model
    from src.constants import PERCENT_T_DICT
    import src.utilities.pickling as pickling

    os.makedirs(os.path.join(output_dir, "visualization"), exist_ok=True)

    params = {
        "device_type": "cpu",
        "cam_size": (46, 30),
        "K": 6,
        "crop_shape": (256, 256),
        "post_processing_dim": 256,
        "num_classes": 2,
        "use_v1_global": False,
        "max_crop_noise": (100, 100),
        "max_crop_size_noise": 100,
        "image_path": cropped_dir,
        "segmentation_path": output_dir,
        "output_path": output_dir,
    }

    exam_list_raw = pickling.unpickle_from_file(pkl_final)
    exam_list = [
        exam for exam in exam_list_raw
        if all(len(exam[v]) > 0 for v in ["L-CC", "R-CC", "L-MLO", "R-MLO"])
    ]
    skipped = len(exam_list_raw) - len(exam_list)
    if skipped > 0:
        print(f"  {skipped} examens ignores (vues manquantes), {len(exam_list)} examens complets")
    if len(exam_list) == 0:
        print("ERREUR : Aucun examen complet (4 vues). Impossible de lancer l'inference.")
        return None

    all_dfs = []

    for idx in ["1", "2", "3", "4", "5"]:
        params["percent_t"] = PERCENT_T_DICT[idx]
        model = gmic_module.GMIC(params)
        model_path = os.path.join(GMIC_DIR, "models", f"sample_model_{idx}.p")
        model.load_state_dict(
            torch.load(model_path, map_location="cpu"), strict=False
        )
        df = run_model(model, exam_list, params, turn_on_visualization=False)
        all_dfs.append(df)
        print(f"  Modele {idx} : {len(df)} predictions")

    pred_df = (
        pd.concat(all_dfs)
        .groupby("image_index")
        .apply(lambda r: pd.Series({
            "malignant_pred":  float(np.nanmean(r["malignant_pred"])),
            "benign_pred":     float(np.nanmean(r["benign_pred"])),
            "malignant_label": int(r.iloc[0]["malignant_label"]),
        }))
        .reset_index()
    )

    pred_df.to_csv(predictions_csv, index=False, float_format="%.4f")
    print(f"\nPredictions sauvegardees : {predictions_csv}")
    return pred_df


# ── Affichage des résultats ───────────────────────────────────────────────────

def show_results(pred_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("RESULTATS")
    print("=" * 60)

    print(f"\n{'Image':35s}  {'Score':>6s}  {'Label':>6s}  {'OK?':>4s}")
    print("-" * 58)
    for _, row in pred_df.iterrows():
        ok = (row["malignant_pred"] > 0.5) == bool(row["malignant_label"])
        label_str = "cancer" if row["malignant_label"] else "sain"
        mark = "OK" if ok else "X"
        print(f"  {row['image_index']:33s}  {row['malignant_pred']:.4f}  {label_str:>6s}  {mark:>4s}")

    y_true = pred_df["malignant_label"].astype(int).tolist()
    y_pred = pred_df["malignant_pred"].tolist()
    y_bin = [int(p > 0.5) for p in y_pred]

    n_cancer = sum(y_true)
    n_sain = len(y_true) - n_cancer
    print(f"\nImages totales : {len(y_true)} ({n_cancer} cancer, {n_sain} saines)")

    if n_cancer > 0 and n_sain > 0:
        from sklearn.metrics import roc_auc_score, classification_report
        auc = roc_auc_score(y_true, y_pred)
        print(f"AUC-ROC (ensemble) : {auc:.4f}")
        print()
        print(classification_report(y_true, y_bin, target_names=["Sain", "Cancer"], zero_division=0))
    else:
        print("(AUC-ROC non calculable : une seule classe presente)")
        correct = sum(t == p for t, p in zip(y_true, y_bin))
        print(f"Accuracy : {correct}/{len(y_true)} = {correct/len(y_true):.2%}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Inference GMIC — Ensemble de 5 modeles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python scripts/inference.py --output-dir output/demo
  python scripts/inference.py --output-dir output/dicom --predictions-csv output/dicom/results.csv
        """,
    )
    parser.add_argument("--output-dir", required=True,
                        help="Dossier de sortie du pretraitement (contient cropped_images/ et data.pkl)")
    parser.add_argument("--predictions-csv", default=None,
                        help="Chemin du CSV de predictions. Par defaut : <output-dir>/predictions.csv")
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    if not os.path.isdir(output_dir):
        print(f"Erreur : dossier introuvable : {output_dir}")
        print("Avez-vous lance le pretraitement ? python scripts/preprocess.py --input-dir ...")
        sys.exit(1)

    cropped_dir = os.path.join(output_dir, "cropped_images")
    pkl_final = os.path.join(output_dir, "data.pkl")

    if not os.path.exists(pkl_final):
        print(f"Erreur : PKL introuvable : {pkl_final}")
        print("Lancez d'abord : python scripts/preprocess.py --input-dir <dir> --output-dir", output_dir)
        sys.exit(1)

    if not os.path.isdir(cropped_dir):
        print(f"Erreur : dossier cropped_images/ introuvable dans {output_dir}")
        sys.exit(1)

    models_dir = os.path.join(GMIC_DIR, "models")
    missing = [f for f in [f"sample_model_{i}.p" for i in range(1, 6)]
               if not os.path.exists(os.path.join(models_dir, f))]
    if missing:
        print(f"Erreur : modeles GMIC manquants dans {models_dir} :")
        for m in missing:
            print(f"  {m}")
        sys.exit(1)

    predictions_csv = args.predictions_csv or os.path.join(output_dir, "predictions.csv")

    pred_df = run_inference(cropped_dir, pkl_final, output_dir, predictions_csv)
    if pred_df is not None:
        show_results(pred_df)
    else:
        print("\nAucune prediction produite (pas d'examens complets).")
        sys.exit(1)


if __name__ == "__main__":
    main()
