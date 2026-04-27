"""
Migration des runs pré-2026-04-24 vers la nouvelle arborescence.

Avant : fine_tuning/checkpoints/runs/{timestamp}_{tag}/
Après : fine_tuning/checkpoints/runs/{target}/{model_tag}/{timestamp}/

- Le target est déduit du `args.json` présent (fallback : "cancer_malignant")
- Un README.md best-effort est généré à partir de l'args.json
  (les champs non tracés dans l'ancien format apparaissent "inconnu")

One-shot, idempotent (skip si le dossier destination existe déjà).
"""

import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fine_tuning.run_metadata import write_run_readme  # noqa: E402

RUNS_DIR = Path(__file__).resolve().parent.parent / "fine_tuning" / "checkpoints" / "runs"

# Format "{YYYYMMDD-HHMMSS}_{tag}"
LEGACY_PATTERN = re.compile(r"^(\d{8}-\d{6})_(.+)$")


def infer_target_and_model_tag(args: dict, tag_suffix: str) -> tuple[str, str]:
    """
    Retourne (target, model_tag) à partir du tag legacy et des args.json.

    Exemples de tag_suffix :
      - "pretrained" → cancer_malignant / resnet18_pretrained
      - "scratch"    → cancer_malignant / resnet18_scratch
      - "normalite_pretrained" / "normalite_scratch"
    """
    target = args.get("target")
    if target is None:
        target = "normalite" if "normalite" in tag_suffix else "cancer_malignant"

    pretrained = args.get("pretrained")
    if pretrained is None:
        pretrained = "pretrained" in tag_suffix

    model_tag = "resnet18_pretrained" if pretrained else "resnet18_scratch"
    return target, model_tag


def enrich_legacy_args(args: dict) -> dict:
    """Complète un args.json legacy pour que write_run_readme puisse tourner."""
    out = dict(args)
    # Champs absents dans l'ancien format — valeurs par défaut / inconnues
    out.setdefault("model_arch", "resnet18 (torchvision)")
    out.setdefault("head_desc", "Linear(512, 1)  (avant l'ajout du Dropout)")
    out.setdefault("dataset_name", "RSNA Breast Cancer Detection (2022, Kaggle)")
    out.setdefault("image_dir", "(legacy — non tracé)")
    out.setdefault("val_split", 0.2)
    out.setdefault("random_seed", 42)
    out.setdefault("aggregation", "par image (chaque vue = un échantillon)")
    out.setdefault("patience", 10)
    out.setdefault("sampler", "WeightedRandomSampler (équilibre classes)")
    out.setdefault("num_workers", "(legacy)")
    out.setdefault("git_commit", "(legacy)")
    out.setdefault("started_at", "(legacy — non tracé)")
    out.setdefault("ended_at", "(legacy — non tracé)")
    # Renommage n_cancer_* / n_anormal_* → n_positive_*
    if "n_cancer_train" in args and "n_positive_train" not in args:
        out["n_positive_train"] = args["n_cancer_train"]
    if "n_cancer_val" in args and "n_positive_val" not in args:
        out["n_positive_val"] = args["n_cancer_val"]
    if "n_anormal_train" in args and "n_positive_train" not in args:
        out["n_positive_train"] = args["n_anormal_train"]
    if "n_anormal_val" in args and "n_positive_val" not in args:
        out["n_positive_val"] = args["n_anormal_val"]
    return out


def best_from_logs(run_dir: Path) -> tuple[float | None, int | None, int]:
    """Retourne (best_auc, best_epoch, epochs_ran) depuis logs.json si présent."""
    logs_file = run_dir / "logs.json"
    if not logs_file.exists():
        return None, None, 0
    logs = json.loads(logs_file.read_text())
    epochs = logs.get("epochs", [])
    if not epochs:
        return None, None, 0
    best = max(epochs, key=lambda e: e.get("auc") or -1)
    return best.get("auc"), best.get("epoch"), len(epochs)


def migrate_one(src: Path) -> None:
    match = LEGACY_PATTERN.match(src.name)
    if not match:
        print(f"  SKIP (pas legacy) : {src.name}")
        return

    timestamp, tag_suffix = match.groups()
    args_file = src / "args.json"
    args = json.loads(args_file.read_text()) if args_file.exists() else {}
    target, model_tag = infer_target_and_model_tag(args, tag_suffix)

    dst = RUNS_DIR / target / model_tag / timestamp
    if dst.exists():
        print(f"  SKIP (deja migre) : {src.name} → {dst.relative_to(RUNS_DIR)}")
        return

    print(f"  MIGRATE : {src.name} → {dst.relative_to(RUNS_DIR)}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))

    # Régénère README best-effort
    enriched = enrich_legacy_args(args)
    enriched["target"] = target
    best_auc, best_epoch, epochs_ran = best_from_logs(dst)
    enriched["best_auc"] = round(best_auc, 4) if best_auc else None
    enriched["best_epoch"] = best_epoch
    enriched["epochs_ran"] = epochs_ran
    enriched["total_time_s"] = None
    enriched["total_time_human"] = "(legacy — non tracé)"
    write_run_readme(dst, enriched)


def main() -> None:
    if not RUNS_DIR.is_dir():
        print(f"Erreur : {RUNS_DIR} introuvable")
        sys.exit(1)

    legacy = [p for p in RUNS_DIR.iterdir() if p.is_dir() and LEGACY_PATTERN.match(p.name)]
    if not legacy:
        print("Aucun run legacy à migrer.")
        return

    print(f"Runs legacy détectés : {len(legacy)}")
    for p in legacy:
        migrate_one(p)
    print("\nMigration terminée.")


if __name__ == "__main__":
    main()
