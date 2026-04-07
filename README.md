# GMIC Breast Cancer Detection

Pipeline de detection du cancer du sein sur mammographies, base sur le modele [GMIC](https://github.com/nyukat/GMIC).

Le projet est maintenant decoupe en deux parties :

- **Pipeline principal GMIC** (preprocess, infer, run, validation, tests) dans `scripts/`.
- **Extraction Kaggle limitee (<= 1 Go)** deplacee dans `extraction_project/` pour la garder a part.

> Documentation detaillee : [`notebooks/pipeline_gmic.qmd`](notebooks/pipeline_gmic.qmd)

---

## Prerequis

- **[Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda** — conda est utilise pour creer un environnement Python isole avec les dependances exactes du projet. Si vous n'avez pas conda, installez Miniconda en premier.
- **Compte Kaggle** avec acces a la competition [rsna-breast-cancer-detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection) et une cle API (kaggle.com → Settings → API → *Create New Token* → telecharge `kaggle.json`).
- **Poids GMIC** (`sample_model_1.p` a `sample_model_5.p`) a placer dans `GMIC/models/`.

---

## Installation

```bash
# 1. Creer l'environnement conda avec toutes les dependances
make build

# 2. Activer l'environnement (a faire dans chaque nouveau terminal)
conda activate gmic

# 3. Configurer les cles Kaggle
make setup
```

`make setup` cree un fichier `.env` depuis `.env.example`. Editez-le et renseignez uniquement vos identifiants Kaggle :

```ini
KAGGLE_USERNAME=votre_username   # depuis kaggle.json → "username"
KAGGLE_KEY=votre_api_key         # depuis kaggle.json → "key"
```

Les commandes `make` refusent de s'executer tant que le fichier `.env` est absent. La commande `make pull-data` exige en plus que `KAGGLE_USERNAME` et `KAGGLE_KEY` soient renseignes (pas les valeurs placeholder).

---

## Utilisation

Les chemins vers vos donnees sont a preciser a chaque commande via `INPUT_DIR` et `OUTPUT_DIR` :

```bash
# (Obligatoire avant pull-data) creer/charger .env
make setup

# Telecharger un subset Kaggle (code extraction deplace dans extraction_project/, conseille pour petits volumes <= 1 Go)
make pull-data

# Verifier les donnees avant de lancer (format images, CSV, vues)
make validate INPUT_DIR=data/demo

# Pipeline complete en une seule commande (preprocess + inference)
make run INPUT_DIR=data/demo OUTPUT_DIR=output/demo

# Pretraitement des images (etapes 1-5)
make preprocess INPUT_DIR=data/demo OUTPUT_DIR=output/demo

# Inference GMIC (etapes 6-7)
make infer OUTPUT_DIR=output/demo
```

> `make pull-data` requiert toujours `make setup` au prealable, car les identifiants Kaggle sont lus depuis `.env`.

---

## Tests

Deux commandes complementaires :

```bash
# Mon code de validation fonctionne-t-il ?
make test                          # tests unitaires (images synthetiques, rapide)

# Mes donnees reelles sont-elles pretes ?
make validate INPUT_DIR=data/demo  # validation de vraies images
```

| Commande | Question | Quoi |
|---|---|---|
| `make test` | "Mon **code** marche ?" | Fabrique des fausses images et verifie que le validateur les detecte |
| `make validate INPUT_DIR=...` | "Mes **donnees** sont bonnes ?" | Verifie vos vraies images (format, taille, CSV) |

### Rapport visuel (notebook)

Le notebook combine les deux dans un rapport HTML :

- **Partie 1** — Tests unitaires : visualise chaque cas synthetique et le verdict du validateur
- **Partie 2** — Vraies images : passe les images de `data/test_images/` dans le validateur

```bash
make notebook NOTEBOOK=test        # genere le rapport HTML
make notebook-serve NOTEBOOK=test  # rapport en live, se rafraichit a chaque sauvegarde du .qmd
```

> Le notebook **re-execute vraiment les tests et la validation** a chaque rendu — il ne lit pas les resultats de `make test`.

---

## Notebooks (Quarto)

Les notebooks de ce projet sont au format `.qmd` ([Quarto](https://quarto.org)) — un format texte (Markdown + blocs de code Python) qui se rend en HTML interactif.

### Deux types de notebooks

| Notebook | Type | Ce qu'il fait |
|---|---|---|
| `test` | **Executeur** | Lance vraiment `pytest` + `check_image()` a chaque rendu — produit les resultats |
| `extract` | **Executeur** | Telecharge vraiment les images depuis Kaggle a chaque rendu (notebook deplace dans `extraction_project/notebook/`) |
| `preprocess` | **Visionneuse** | Lit ce que `make preprocess` a deja produit dans `OUTPUT_DIR` — n'execute rien |
| `pipeline` | **Visionneuse** | Lit ce que `make infer` a deja produit dans `OUTPUT_DIR` — n'execute rien |

> Les notebooks **visionneuses** (`preprocess`, `pipeline`) ne montrent rien si le dossier `OUTPUT_DIR` est vide. Il faut d'abord lancer le pipeline.

### Commandes

```bash
# Generer un HTML statique (pour partager ou archiver)
make notebook NOTEBOOK=test                               # rapport de tests (s'execute seul)
make notebook NOTEBOOK=extract                            # extraction Kaggle (code dans extraction_project/)
make notebook NOTEBOOK=preprocess OUTPUT_DIR=output/demo  # diagnostic pretraitement (necessite OUTPUT_DIR)
make notebook NOTEBOOK=pipeline   OUTPUT_DIR=output/demo  # inspection predictions (necessite OUTPUT_DIR)

# Previsualiser en live dans le navigateur (re-execute a chaque sauvegarde du .qmd)
make notebook-serve NOTEBOOK=test
make notebook-serve NOTEBOOK=preprocess OUTPUT_DIR=output/demo
```

> `make notebook-serve` utilise `quarto preview` : il re-execute tout le notebook et rafraichit le navigateur automatiquement a chaque modification du fichier `.qmd`.

> Les fichiers `.html` generes sont ignores par git (voir `.gitignore`). Pour les partager, envoyez directement le `.html` produit.

---

## Flux de donnees

```mermaid
flowchart LR
    A([Kaggle\nRSNA dataset]) -->|make pull-data| B[data/\ntrain_images/\ntrain.csv]

    B --> C{Format ?}
    C -->|DICOM .dcm| D[Conversion\nDICOM -> PNG]
    C -->|PNG| E[Construction\ndu PKL GMIC]
    D --> E

    E --> F[Recadrage\ncrop_mammogram]
    F --> G[Resize 2944x1920\n+ normalisation uint8]
    G --> H[Centres optimaux\nget_optimal_centers]
    H --> I[Inference\n5 modeles GMIC]

    I --> J([output/\npredictions.csv\nAUC-ROC])
```

---

## Commandes disponibles

| Commande | Description |
|---|---|
| `make build` | Creer l'environnement et installer les dependances |
| `make setup` | Configurer `.env`, verifier Kaggle |
| `make check` | Verifier dependances, modeles GMIC, donnees |
| `make pull-data` | Telecharger les donnees depuis Kaggle |
| `make validate` | Valider les donnees d'entree (format, CSV, images) |
| `make run` | Lancer la pipeline complete GMIC en une seule commande |
| `make preprocess` | Lancer uniquement le pretraitement (etapes 1-5) |
| `make infer` | Lancer uniquement l'inference (etapes 6-7) |
| `make notebook [NOTEBOOK=...]` | Rendre un notebook en HTML (pipeline, test, extract, preprocess) |
| `make notebook-serve [NOTEBOOK=...]` | Servir un notebook HTML en local (http://localhost:8080) |
| `make freeze` | Figer les versions des packages |
| `make test` | Lancer les tests unitaires |

Options avancees :

```bash
# Forcer une etape specifique
make preprocess INPUT_DIR=data/demo OUTPUT_DIR=output/demo ARGS="--force-crop"
make preprocess INPUT_DIR=data/demo OUTPUT_DIR=output/demo ARGS="--force-resize"

# Format explicite
make preprocess INPUT_DIR=data/demo OUTPUT_DIR=output/demo ARGS="--format png"

# Predictions dans un fichier specifique
make infer OUTPUT_DIR=output/demo ARGS="--predictions-csv output/demo/preds.csv"
```

---

## Structure du projet

```
.
├── Makefile
├── pyproject.toml
├── environment.yml           <- Environnement conda (make build)
├── .env.example              <- Copier en .env et remplir
├── GMIC/                     <- Modele GMIC (poids dans GMIC/models/)
├── extraction_project/       <- Projet extraction Kaggle (petits volumes <= 1 Go)
│   ├── script/
│   │   └── extract_download.py        <- Telechargement Kaggle (anti-429)
│   └── notebook/
│       └── extract_download.qmd       <- Notebook extraction Kaggle
├── scripts/
│   ├── run_gmic_pipeline.py  <- Pipeline complet (make run)
│   ├── preprocess.py         <- Pretraitement GMIC (etapes 1-5)
│   ├── inference.py          <- Inference GMIC (etapes 6-7)
│   └── validate_input.py     <- Validation des donnees d'entree
├── notebooks/
│   ├── pipeline_gmic.qmd           <- Inspection des sorties presentes dans output/
│   ├── preprocess_gmic.qmd         <- Diagnostic du pretraitement
│   └── test_validation_report.qmd  <- Rapport de tests avec images
├── tests/
│   └── test_validate_input.py  <- Tests unitaires (CSV, images)
├── doc/
│   └── troubleshooting.md     <- Erreurs courantes et solutions
├── data/                     <- Images + CSV (gitignore, a remplir)
│   └── test_images/          <- Images de test (mauvais formats)
└── output/                   <- Resultats (genere par make preprocess/make infer)
```

---

## Limitations

- **Domain shift** : GMIC est entraine sur INbreast/CBIS-DDSM — les performances sur RSNA sont inferieures aux scores publies (AUC ~0.87 sur donnees d'origine)
- **Au minimum 1 image par patient** : GMIC analyse chaque image individuellement, pas de contrainte sur le nombre de vues
- **CPU only** par defaut
