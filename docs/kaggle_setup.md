# Télécharger les données depuis Kaggle

Ce guide explique comment configurer l'accès Kaggle et télécharger les images
de la competition [RSNA Breast Cancer Detection](https://www.kaggle.com/competitions/rsna-breast-cancer-detection).

---

## Prérequis

- Un **compte Kaggle** (gratuit) : [kaggle.com/account/login](https://www.kaggle.com/account/login)
- Avoir **accepté les règles** de la compétition RSNA sur Kaggle (obligatoire pour accéder aux données)
- Une **clé API** Kaggle : `kaggle.json`

---

## 1. Obtenir votre clé API Kaggle

1. Connectez-vous sur [kaggle.com](https://www.kaggle.com)
2. Cliquez sur votre avatar → **Settings**
3. Section **API** → **Create New Token**
4. Un fichier `kaggle.json` est téléchargé, contenant :

```json
{"username": "votre_username", "key": "votre_api_key"}
```

---

## 2. Configurer le projet

```bash
make setup
```

Cette commande crée un fichier `.env` depuis `.env.example`. Éditez-le avec vos identifiants :

```ini
KAGGLE_USERNAME=votre_username   # champ "username" de kaggle.json
KAGGLE_KEY=votre_api_key         # champ "key" de kaggle.json
```

Vérifiez que la configuration est correcte :

```bash
make check
```

---

## 3. Télécharger les données

```bash
make pull-data
```

Ce script (`extraction_project/script/extract_download.py`) télécharge un sous-ensemble
de la compétition RSNA avec une gestion automatique des erreurs 429 (rate limiting Kaggle).

Il produit dans `data/` :
```
data/
├── train_images/    <- images DICOM organisées par patient
└── train.csv        <- métadonnées (patient_id, label cancer, latéralité, vue)
```

---

## 4. Lancer le pipeline

Une fois les données téléchargées :

```bash
make run INPUT_DIR=data OUTPUT_DIR=preprocess_image/rsna
```

---

## Gestion du rate limiting (erreur 429)

Kaggle limite le nombre de requêtes simultanées. Le script `extract_download.py`
inclut un mécanisme de retry avec backoff exponentiel. Si vous obtenez quand même
des erreurs 429, relancez simplement `make pull-data` — le script reprend là où il s'est arrêté.

---

## Notebook d'extraction

Pour visualiser et contrôler le téléchargement interactivement :

```bash
make notebook NOTEBOOK=extract
```

Ce notebook (`extraction_project/notebook/extract_download.qmd`) permet de configurer
le nombre de patients à télécharger et de suivre la progression.
