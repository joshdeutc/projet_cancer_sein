"""
Téléchargement intelligent depuis Kaggle — RSNA Breast Cancer
-------------------------------------------------------------
Télécharge un sous-ensemble stratifié d'images DICOM depuis la
compétition Kaggle RSNA Breast Cancer Detection.

Améliorations :
  - Pas de token en dur (utilise ~/.kaggle/kaggle.json ou KAGGLE_USERNAME/KAGGLE_KEY)
  - Rate limiting AIMD (×1.5 sur 429, −0.5s après 50 succès)
  - Auto-décompression des .zip après téléchargement
  - Valeurs par défaut RSNA (Entrée pour accepter)
  - Compteurs downloaded/skipped/failed en temps réel
  - Reprise automatique (skip les fichiers déjà présents)
  - Mode cooldown après gros Retry-After (>120s)

Usage :
  python scripts/extract_download.py
"""

import os
import random
import time
import zipfile

import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi
from requests.exceptions import HTTPError


# ── Configuration ─────────────────────────────────────────────────────────────

COMPETITION = "rsna-breast-cancer-detection"

# Valeurs par défaut RSNA (l'utilisateur appuie sur Entrée pour accepter)
DEFAULTS = {
    "base_dir":    "extract_dataset",
    "target_col":  "cancer",
    "group_col":   "patient_id",
    "percentage":  "5",
}

# Rate limiting AIMD
PAUSE_MIN       = 10     # plancher absolu (secondes)
PAUSE_INITIAL   = 20     # pause de départ
PAUSE_MAX       = 120    # plafond (au-delà, on attend juste Retry-After)
MULT_INCREASE   = 1.5    # ×1.5 sur 429
ADDITIVE_DECREASE = 0.5  # −0.5s après OK_BEFORE_PROBE succès
OK_BEFORE_PROBE = 50     # succès consécutifs avant de sonder plus bas
MAX_RETRIES     = 3      # tentatives max par fichier


# ── Helpers ───────────────────────────────────────────────────────────────────

def prompt_default(message: str, default: str) -> str:
    """Prompt interactif avec valeur par défaut entre crochets."""
    result = input(f"{message} [{default}]: ").strip()
    return result if result else default


def unzip_and_clean(filepath: str) -> bool:
    """Décompresse <filepath>.zip (si présent), supprime l'archive. Retourne True si dézippé."""
    zip_path = filepath if filepath.endswith(".zip") else filepath + ".zip"
    if not os.path.exists(zip_path):
        return False
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(os.path.dirname(zip_path))
        os.remove(zip_path)
        return True
    except zipfile.BadZipFile:
        return False


def is_present(task: dict) -> bool:
    """Vérifie si le fichier (ou sa version .zip) existe déjà."""
    return os.path.exists(task["local_file"]) or os.path.exists(task["local_file"] + ".zip")


# ── Authentification ──────────────────────────────────────────────────────────

def authenticate() -> KaggleApi:
    """Authentification via ~/.kaggle/kaggle.json ou variables d'environnement."""
    api = KaggleApi()
    api.authenticate()
    return api


# ── Récupération du CSV ──────────────────────────────────────────────────────

def ensure_csv(api: KaggleApi, data_dir: str) -> str:
    """S'assure que train.csv est disponible localement."""
    zip_path = os.path.join(data_dir, "train.csv.zip")
    csv_path = os.path.join(data_dir, "train.csv")

    # Dézipper si besoin
    if os.path.exists(zip_path) and not os.path.exists(csv_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        print("  train.csv dézippé.")

    # Télécharger si absent
    if not os.path.exists(csv_path):
        print("  Téléchargement de train.csv depuis Kaggle...")
        api.competition_download_file(
            competition=COMPETITION,
            file_name="train.csv",
            path=data_dir,
            force=False,
            quiet=False,
        )
        if os.path.exists(zip_path) and not os.path.exists(csv_path):
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(data_dir)
        if not os.path.exists(csv_path):
            print("  ERREUR : téléchargement de train.csv échoué.")
            exit(1)

    return csv_path


# ── Configuration interactive ─────────────────────────────────────────────────

def interactive_config(api: KaggleApi) -> dict:
    """Configuration interactive avec valeurs par défaut RSNA."""

    print(f"\nInspection de la compétition : {COMPETITION}\n")
    api.competition_list_files_cli(COMPETITION)

    # Dossier
    base_dir = prompt_default("\nDossier de destination", DEFAULTS["base_dir"])
    data_dir = os.path.join("data", base_dir)
    os.makedirs(data_dir, exist_ok=True)

    # CSV
    csv_path = ensure_csv(api, data_dir)
    df = pd.read_csv(csv_path)

    # Colonne cible
    target_col = prompt_default("Colonne cible", DEFAULTS["target_col"])
    while target_col not in df.columns:
        print(f"  Introuvable. Colonnes : {list(df.columns)}")
        target_col = input("Colonne cible : ").strip()

    # Colonne patient
    group_col = prompt_default("Colonne patient", DEFAULTS["group_col"])
    while group_col not in df.columns:
        print(f"  Introuvable. Colonnes : {list(df.columns)}")
        group_col = input("Colonne patient : ").strip()

    # Pourcentage
    while True:
        try:
            pct_str = prompt_default("Pourcentage du dataset (1-100)", DEFAULTS["percentage"])
            percentage = float(pct_str) / 100.0
            if 0 < percentage <= 1.0:
                break
            print("  Entrez un nombre entre 1 et 100.")
        except ValueError:
            print("  Entrée invalide.")

    return {
        "data_dir": data_dir,
        "df": df,
        "target_col": target_col,
        "group_col": group_col,
        "percentage": percentage,
    }


# ── Sélection stratifiée ─────────────────────────────────────────────────────

def build_subset(df: pd.DataFrame, target_col: str, group_col: str,
                 percentage: float) -> pd.DataFrame:
    """Construit un sous-ensemble stratifié du dataset."""

    target_data = df[target_col].dropna()
    num_unique = len(target_data.unique())
    is_multi_label = target_data.astype(str).str.contains(r",|\|").any()
    can_stratify = not (is_multi_label or num_unique > 50)

    if can_stratify:
        print(f"  {num_unique} classes — stratification activée.")
    else:
        print("  Multi-label / continu — split aléatoire.")

    grouped = df.groupby(group_col)[target_col].max().reset_index()

    if can_stratify:
        selected, _ = train_test_split(
            grouped, train_size=percentage,
            stratify=grouped[target_col], random_state=42,
        )
    else:
        selected = grouped.sample(frac=percentage, random_state=42)

    df_subset = df[df[group_col].isin(selected[group_col])]

    # Garantir ≥1 positif et ≥1 négatif
    if can_stratify:
        selected_ids = set(selected[group_col])
        for label, name in [(1, "positif"), (0, "négatif")]:
            pool = grouped[grouped[target_col] == label]
            if not any(pid in selected_ids for pid in pool[group_col]):
                forced = pool.sample(1, random_state=42)
                print(f"  Ajout forcé d'un {name} : patient {forced[group_col].values[0]}")
                selected = pd.concat([selected, forced], ignore_index=True)
                selected_ids = set(selected[group_col])
                df_subset = df[df[group_col].isin(selected_ids)]

    print(f"\n  Distribution :")
    for val, count in df_subset[target_col].value_counts().items():
        print(f"    {target_col}={val} : {count}")

    return df_subset


# ── Préparation des tâches ────────────────────────────────────────────────────

def prepare_tasks(df_subset: pd.DataFrame, group_col: str, data_dir: str) -> list:
    """Prépare la liste des fichiers à télécharger."""
    tasks = []
    for _, row in df_subset.iterrows():
        pid = str(row[group_col])
        iid = str(row["image_id"])
        local_dir = os.path.join(data_dir, "train_images", pid)
        tasks.append({
            "kaggle_path": f"train_images/{pid}/{iid}.dcm",
            "local_dir":   local_dir,
            "local_file":  os.path.join(local_dir, f"{iid}.dcm"),
        })
    return tasks


# ── Rate limiter AIMD ─────────────────────────────────────────────────────────

class RateLimiter:
    """
    Additive Increase / Multiplicative Decrease — inspiré du contrôle de
    congestion TCP.

    - Succès : après OK_BEFORE_PROBE succès consécutifs, pause −0.5s (additif)
    - 429    : pause ×1.5 (multiplicatif), plancher relevé
    - Gros Retry-After (≥120s) : mode cooldown temporaire (×1.5 du plancher)
    """

    def __init__(self):
        self.pause = PAUSE_INITIAL
        self.floor = PAUSE_INITIAL
        self.consecutive_ok = 0
        self.total_429s = 0
        self.cooldown_until = 0.0

    def wait(self) -> float:
        """Attend avant la prochaine requête. Retourne le temps attendu."""
        effective = self.pause
        if time.time() < self.cooldown_until:
            effective = max(self.pause, self.floor * 1.5)
        jitter = random.uniform(0.5, 2.0)
        total = effective + jitter
        time.sleep(total)
        return total

    def on_success(self) -> str | None:
        """Appelé après un succès. Retourne un message si le plancher change."""
        self.consecutive_ok += 1

        if self.consecutive_ok >= OK_BEFORE_PROBE:
            old = self.floor
            self.floor = max(PAUSE_MIN, self.floor - ADDITIVE_DECREASE)
            self.pause = self.floor
            self.consecutive_ok = 0
            if self.floor < old:
                return f"sonde {old:.0f}s → {self.floor:.0f}s"
            return None

        # Redescendre doucement vers le plancher après une série de succès
        if self.consecutive_ok >= 10 and self.pause > self.floor:
            self.pause = max(self.floor, self.pause - 1)

        return None

    def on_429(self, retry_after: str | None = None) -> float:
        """
        Appelé sur la PREMIÈRE 429 d'un fichier.
        Augmente le plancher (×1.5) et retourne le temps d'attente.
        """
        self.consecutive_ok = 0
        self.total_429s += 1

        old = self.floor
        self.floor = min(PAUSE_MAX, self.floor * MULT_INCREASE)
        self.pause = self.floor

        base = int(retry_after) if retry_after else 60

        # Gros Retry-After → cooldown : rester conservateur 60s de plus
        if base >= 120:
            self.cooldown_until = time.time() + base + 60

        wait = base + random.uniform(2, 8)
        tqdm.write(
            f"  429 — plancher {old:.0f}s → {self.floor:.0f}s, "
            f"attente {wait:.0f}s (Retry-After={base}s)"
        )
        return wait

    def status(self) -> str:
        return f"floor={self.floor:.0f}s ok={self.consecutive_ok} 429s={self.total_429s}"


# ── Boucle de téléchargement ─────────────────────────────────────────────────

def download_all(api: KaggleApi, tasks: list) -> dict:
    """Télécharge les fichiers avec rate limiting AIMD."""

    limiter = RateLimiter()
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "unzipped": 0}

    # Pré-filtrer les fichiers déjà présents
    remaining = []
    for task in tasks:
        if is_present(task):
            stats["skipped"] += 1
            if unzip_and_clean(task["local_file"]):
                stats["unzipped"] += 1
        else:
            remaining.append(task)

    if stats["skipped"]:
        print(f"  {stats['skipped']} fichiers déjà présents (skip)")
    if not remaining:
        print("  Rien à télécharger !")
        return stats

    print(f"  {len(remaining)} fichiers à télécharger\n")

    pbar = tqdm(
        remaining, desc="Téléchargement", unit="img",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )

    for task in pbar:
        os.makedirs(task["local_dir"], exist_ok=True)

        # Vérification de reprise (un autre processus a pu finir entre-temps)
        if is_present(task):
            stats["skipped"] += 1
            if unzip_and_clean(task["local_file"]):
                stats["unzipped"] += 1
            continue

        limiter.wait()
        pbar.set_postfix_str(limiter.status())

        success = False
        attempts = 0

        while attempts <= MAX_RETRIES and not success:
            try:
                api.competition_download_file(
                    competition=COMPETITION,
                    file_name=task["kaggle_path"],
                    path=task["local_dir"],
                    force=False,
                    quiet=True,
                )
                success = True
                stats["downloaded"] += 1

                if unzip_and_clean(task["local_file"]):
                    stats["unzipped"] += 1

                msg = limiter.on_success()
                if msg:
                    tqdm.write(f"  [{stats['downloaded']}/{len(remaining)}] {msg}")

            except HTTPError as e:
                status_code = getattr(getattr(e, "response", None), "status_code", None)
                if status_code == 429:
                    retry_after = e.response.headers.get("Retry-After")

                    # Seulement la première 429 par fichier ajuste le plancher
                    if attempts == 0:
                        wait = limiter.on_429(retry_after)
                    else:
                        base = int(retry_after) if retry_after else 60
                        wait = base + random.uniform(2, 8)
                        tqdm.write(f"    retry {attempts + 1}/{MAX_RETRIES + 1} dans {wait:.0f}s")

                    attempts += 1
                    time.sleep(wait)
                else:
                    tqdm.write(f"  Erreur HTTP sur {task['kaggle_path']}: {e}")
                    stats["failed"] += 1
                    break

            except Exception as e:
                tqdm.write(f"  Erreur sur {task['kaggle_path']}: {e}")
                stats["failed"] += 1
                break

        if not success:
            stats["failed"] += 1

    return stats


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  Kaggle RSNA Breast Cancer — Téléchargement intelligent")
    print("=" * 58)

    api = authenticate()
    config = interactive_config(api)

    # Subset stratifié
    print(f"\nConstruction du subset ({config['percentage'] * 100:.1f}%)...")
    df_subset = build_subset(
        config["df"], config["target_col"],
        config["group_col"], config["percentage"],
    )

    # Sauvegarder le CSV subset
    subset_csv = os.path.join(config["data_dir"], "train_subset.csv")
    df_subset.to_csv(subset_csv, index=False)
    print(f"\n  CSV subset : {subset_csv}")

    # Préparer les tâches
    tasks = prepare_tasks(df_subset, config["group_col"], config["data_dir"])
    n = len(tasks)
    print(f"\n{'=' * 58}")
    print(f"  {n} images (~{n * 5 / 1024:.1f} Go estimés)")
    print(f"{'=' * 58}")

    if input("\nDémarrer le téléchargement ? (y/n): ").lower() not in ("y", "yes"):
        print("Annulé.")
        return

    print("\nDémarrage du téléchargement (AIMD rate limiting)...\n")
    stats = download_all(api, tasks)

    # Résumé
    print(f"\n{'=' * 58}")
    print(f"  TERMINÉ")
    print(f"    Téléchargés  : {stats['downloaded']}")
    print(f"    Déjà présents: {stats['skipped']}")
    print(f"    Dézippés     : {stats['unzipped']}")
    print(f"    Échoués      : {stats['failed']}")
    total_429 = stats.get("total_429s", 0)
    print(f"    429 reçus    : {total_429}")
    print(f"  Images : {os.path.join(config['data_dir'], 'train_images')}")
    print(f"{'=' * 58}")


if __name__ == "__main__":
    main()
