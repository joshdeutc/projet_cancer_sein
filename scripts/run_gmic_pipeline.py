"""
Pipeline GMIC unifie - wrapper fin
----------------------------------
Orchestre les deux etapes du pipeline :
  1) preprocess.py (etapes 1-5)
  2) inference.py  (etapes 6-7)

Usage :
  python scripts/run_gmic_pipeline.py --input-dir data/demo --output-dir output/demo
  python scripts/run_gmic_pipeline.py --input-dir data/extract_dataset --force-centers
"""

import argparse
import os
import subprocess
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)


def _run_step(cmd: list[str], step_name: str) -> None:
    print("\n" + "=" * 60)
    print(step_name)
    print("=" * 60)
    print("Commande:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=PROJECT_DIR)
    if result.returncode != 0:
        print(f"\nERREUR: {step_name} a echoue (code {result.returncode})")
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline GMIC unifie - wrapper preprocess + inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python scripts/run_gmic_pipeline.py --input-dir data/demo --output-dir output/demo
  python scripts/run_gmic_pipeline.py --input-dir data/extract_dataset --force-crop
        """,
    )
    parser.add_argument("--input-dir", required=True,
                        help="Dossier contenant les images (DICOM ou PNG)")
    parser.add_argument("--csv", default=None,
                        help="Chemin vers le CSV. Defaut : detecte depuis input-dir")
    parser.add_argument("--output-dir", default=None,
                        help="Dossier de sortie. Defaut : <projet>/output/")
    parser.add_argument("--format", choices=["dicom", "png", "auto"], default="auto",
                        help="Format des images d'entree (defaut: auto-detection)")
    parser.add_argument("--force-crop", action="store_true",
                        help="Forcer le crop")
    parser.add_argument("--force-resize", action="store_true",
                        help="Forcer le resize")
    parser.add_argument("--force-centers", action="store_true",
                        help="Forcer get_optimal_centers")
    parser.add_argument("--predictions-csv", default=None,
                        help="Chemin du CSV de predictions. Defaut : <output-dir>/predictions.csv")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    if not os.path.isdir(input_dir):
        print(f"Erreur : dossier introuvable : {input_dir}")
        sys.exit(1)

    output_dir = os.path.abspath(args.output_dir) if args.output_dir else os.path.join(PROJECT_DIR, "output")

    preprocess_cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "preprocess.py"),
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--format",
        args.format,
    ]
    if args.csv:
        preprocess_cmd.extend(["--csv", args.csv])
    if args.force_crop:
        preprocess_cmd.append("--force-crop")
    if args.force_resize:
        preprocess_cmd.append("--force-resize")
    if args.force_centers:
        preprocess_cmd.append("--force-centers")

    inference_cmd = [
        sys.executable,
        os.path.join(SCRIPT_DIR, "inference.py"),
        "--output-dir",
        output_dir,
    ]
    if args.predictions_csv:
        inference_cmd.extend(["--predictions-csv", args.predictions_csv])

    _run_step(preprocess_cmd, "ETAPE 1/2 : PREPROCESS")
    _run_step(inference_cmd, "ETAPE 2/2 : INFERENCE")

    print("\n" + "=" * 60)
    print("PIPELINE TERMINE")
    print(f"  Output dir      : {output_dir}")
    print(f"  Predictions CSV : {args.predictions_csv or os.path.join(output_dir, 'predictions.csv')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
