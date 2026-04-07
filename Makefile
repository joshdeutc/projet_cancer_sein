## Pipeline GMIC - Detection du cancer du sein
## Usage : make <commande>

.DEFAULT_GOAL := help
.PHONY: help build setup check validate pull-data preprocess infer notebook notebook-serve run freeze test

CONDA_ENV  := gmic
CONDA_RUN  := conda run -n $(CONDA_ENV) --no-capture-output
NOTEBOOK      ?= pipeline
NOTEBOOK_PORT ?= 8080
NOTEBOOK_KERNEL ?= gmic

# Charger .env si present
-include .env
export

## ── Couleurs ──────────────────────────────────────────────────────────────────
RESET  := \033[0m
BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m

help: ## Affiche cette aide
	@echo ""
	@echo "$(BOLD)Pipeline GMIC - Detection du cancer du sein$(RESET)"
	@echo ""
	@echo "  $(YELLOW)Premiere utilisation :$(RESET)"
	@echo "    make build          <- creer l'environnement conda '$(CONDA_ENV)'"
	@echo "    conda activate $(CONDA_ENV)"
	@echo "    make setup          <- configurer .env (cles Kaggle)"
	@echo "    make pull-data      <- telecharger les images"
	@echo "    make preprocess     <- lancer les etapes 1-5"
	@echo "    make infer          <- lancer les etapes 6-7"
	@echo "    make notebook       <- executer et rendre le notebook en HTML"
	@echo "    make notebook-serve <- visualiser le notebook dans le navigateur"
	@echo "    make run            <- lancer le pipeline"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-15s$(RESET) %s\n", $$1, $$2}'
	@echo ""

build: ## Creer l'environnement conda 'gmic' depuis environment.yml
	@echo "$(YELLOW)Creation de l'environnement conda '$(CONDA_ENV)'...$(RESET)"
	@if conda env list | grep -q "^$(CONDA_ENV) "; then \
		echo "L'environnement '$(CONDA_ENV)' existe deja. Mise a jour..."; \
		conda env update -n $(CONDA_ENV) -f environment.yml --prune; \
	else \
		conda env create -f environment.yml; \
	fi
	@echo ""
	@echo "$(GREEN)Environnement pret.$(RESET)"
	@echo "$(YELLOW)Activez-le avec : conda activate $(CONDA_ENV)$(RESET)"

setup: ## Configurer le projet (variables .env, verifier Kaggle)
	@echo "$(YELLOW)Configuration du projet...$(RESET)"
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN).env cree depuis .env.example - editez-le avec vos valeurs.$(RESET)"; \
	else \
		echo ".env deja present."; \
	fi
	@echo ""
	@echo "Variables de configuration actuelles :"
	@grep -v '^#' .env | grep -v '^$$' | sed 's/^/  /'
	@echo ""
	@if [ -z "$$KAGGLE_USERNAME" ] && [ ! -f ~/.kaggle/kaggle.json ]; then \
		echo "$(YELLOW)ATTENTION : Kaggle non configure. Editez .env avec KAGGLE_USERNAME et KAGGLE_KEY.$(RESET)"; \
	else \
		echo "$(GREEN)Kaggle : OK$(RESET)"; \
	fi

check: ## Verifier que tout est en place avant de lancer le pipeline
	@echo "$(YELLOW)Verification de l'environnement...$(RESET)"
	@echo ""
	@if conda env list | grep -q "^$(CONDA_ENV) "; then \
		echo "  Env conda '$(CONDA_ENV)' : $(GREEN)OK$(RESET)"; \
	else \
		echo "  Env conda '$(CONDA_ENV)' : $(YELLOW)ABSENT - lancez : make build$(RESET)"; \
	fi
	@$(CONDA_RUN) python -c "import cv2, pydicom, torch, sklearn, imageio; print('  Dependances Python : $(GREEN)OK$(RESET)')" 2>/dev/null || \
		echo "  $(YELLOW)Dependances manquantes - lancez : make build$(RESET)"
	@if [ -d "GMIC/models" ] && ls GMIC/models/sample_model_*.p > /dev/null 2>&1; then \
		echo "  Modeles GMIC       : $(GREEN)OK$(RESET) ($(shell ls GMIC/models/sample_model_*.p 2>/dev/null | wc -l) modeles)"; \
	else \
		echo "  $(YELLOW)Modeles GMIC       : MANQUANTS - placez sample_model_1.p..5.p dans GMIC/models/$(RESET)"; \
	fi
	@if [ -d "data" ] && [ "$$(ls data/ 2>/dev/null | wc -l)" -gt 0 ]; then \
		echo "  Dossier data/      : $(GREEN)OK$(RESET)"; \
	else \
		echo "  $(YELLOW)Dossier data/      : VIDE - lancez : make pull-data$(RESET)"; \
	fi

validate: ## Valider les donnees d'entree (images, CSV, format)
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "$(YELLOW)Usage : make validate INPUT_DIR=data/demo$(RESET)"; exit 1; \
	fi
	@$(CONDA_RUN) python scripts/validate_input.py \
		--input-dir $(INPUT_DIR) \
		$(ARGS)

pull-data: ## Telecharger les donnees depuis Kaggle (interactif)
	@echo "$(YELLOW)Lancement du telechargement Kaggle...$(RESET)"
	@$(CONDA_RUN) python scripts/extract_download.py

preprocess: ## Lancer uniquement le pretraitement (etapes 1-5)
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "$(YELLOW)Erreur : INPUT_DIR non defini.$(RESET)"; \
		echo "Usage : make preprocess INPUT_DIR=data/demo OUTPUT_DIR=output/demo"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Lancement du pretraitement GMIC...$(RESET)"
	@echo "  INPUT_DIR  = $(INPUT_DIR)"
	@echo "  OUTPUT_DIR = $(OUTPUT_DIR)"
	@echo ""
	@$(CONDA_RUN) python scripts/preprocess.py \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		$(ARGS)

infer: ## Lancer uniquement l'inference (etapes 6-7)
	@if [ -z "$(OUTPUT_DIR)" ]; then \
		echo "$(YELLOW)Erreur : OUTPUT_DIR non defini.$(RESET)"; \
		echo "Usage : make infer OUTPUT_DIR=output/demo"; \
		exit 1; \
	fi
	@echo "$(YELLOW)Lancement de l'inference GMIC...$(RESET)"
	@echo "  OUTPUT_DIR = $(OUTPUT_DIR)"
	@echo ""
	@$(CONDA_RUN) python scripts/inference.py \
		--output-dir $(OUTPUT_DIR) \
		$(ARGS)

notebook: ## Rendre un notebook en HTML  [NOTEBOOK=pipeline|test|extract|preprocess]
	@QMD=$$(case "$(NOTEBOOK)" in \
		pipeline)   echo "notebooks/pipeline_gmic.qmd" ;; \
		test)       echo "notebooks/test_validation_report.qmd" ;; \
		extract)    echo "notebooks/extract_download.qmd" ;; \
		preprocess) echo "notebooks/preprocess_gmic.qmd" ;; \
		*)          echo "$(NOTEBOOK)" ;; \
	esac); \
	HTML=$${QMD%.qmd}.html; \
	if ! command -v quarto >/dev/null 2>&1; then \
		echo "$(YELLOW)Quarto introuvable. Installez-le : https://quarto.org/docs/get-started/$(RESET)"; exit 1; \
	fi; \
	if [ ! -f "$$QMD" ]; then \
		echo "$(YELLOW)Notebook introuvable : $$QMD$(RESET)"; \
		echo "Valeurs acceptees : pipeline, test, extract, preprocess (ou chemin direct)"; exit 1; \
	fi; \
	$(CONDA_RUN) python -c "import ipykernel, nbformat" >/dev/null 2>&1 || \
		$(CONDA_RUN) python -m pip install -q ipykernel nbformat; \
	$(CONDA_RUN) python -m ipykernel install --user --name $(NOTEBOOK_KERNEL) \
		--display-name "Python ($(CONDA_ENV))" >/dev/null; \
	PYTHON_BIN=$$(conda run -n $(CONDA_ENV) python -c "import sys; print(sys.executable)"); \
	echo "$(YELLOW)Rendu du notebook : $$QMD$(RESET)"; \
	GMIC_OUTPUT_DIR=$(OUTPUT_DIR) QUARTO_PYTHON=$$PYTHON_BIN quarto render $$QMD \
		--to html --execute --no-execute-daemon; \
	echo "$(GREEN)Notebook genere : $$HTML$(RESET)"

notebook-serve: ## Re-executer un notebook et le servir en live (quarto preview)  [NOTEBOOK=pipeline|test|...]
	@QMD=$$(case "$(NOTEBOOK)" in \
		pipeline)   echo "notebooks/pipeline_gmic.qmd" ;; \
		test)       echo "notebooks/test_validation_report.qmd" ;; \
		extract)    echo "notebooks/extract_download.qmd" ;; \
		preprocess) echo "notebooks/preprocess_gmic.qmd" ;; \
		*)          echo "$(NOTEBOOK)" ;; \
	esac); \
	if [ ! -f "$$QMD" ]; then \
		echo "$(YELLOW)Notebook introuvable : $$QMD$(RESET)"; exit 1; \
	fi; \
	$(CONDA_RUN) python -m ipykernel install --user --name $(NOTEBOOK_KERNEL) \
		--display-name "Python ($(CONDA_ENV))" >/dev/null; \
	PYTHON_BIN=$$(conda run -n $(CONDA_ENV) python -c "import sys; print(sys.executable)"); \
	echo "$(YELLOW)Ouvrez dans votre navigateur : http://localhost:$(NOTEBOOK_PORT)/$(RESET)"; \
	echo "$(YELLOW)Re-execution automatique a chaque modification du .qmd$(RESET)"; \
	QUARTO_PYTHON=$$PYTHON_BIN quarto preview $$QMD --port $(NOTEBOOK_PORT) --no-browser

run: ## Lancer le pipeline complet (DICOM ou PNG auto-detecte)
	@if [ -z "$(INPUT_DIR)" ]; then \
		echo "$(YELLOW)Erreur : INPUT_DIR non defini.$(RESET)"; \
		echo ""; \
		echo "  Option 1 — en ligne de commande :"; \
		echo "    make run INPUT_DIR=data/demo OUTPUT_DIR=output/demo"; \
		echo ""; \
		echo "  Option 2 — dans .env (permanent) :"; \
		echo "    echo 'INPUT_DIR=data/demo' >> .env"; \
		echo "    echo 'OUTPUT_DIR=output/demo' >> .env"; \
		echo "    make run"; \
		echo ""; \
		exit 1; \
	fi
	@echo "$(YELLOW)Lancement du pipeline GMIC...$(RESET)"
	@echo "  INPUT_DIR  = $(INPUT_DIR)"
	@echo "  OUTPUT_DIR = $(OUTPUT_DIR)"
	@echo ""
	@$(CONDA_RUN) python scripts/run_gmic_pipeline.py \
		--input-dir $(INPUT_DIR) \
		--output-dir $(OUTPUT_DIR) \
		$(ARGS)

freeze: ## Exporter l'environnement conda exact (environment.lock.yml)
	@echo "$(YELLOW)Export de l'environnement conda...$(RESET)"
	@conda env export -n $(CONDA_ENV) --no-builds > environment.lock.yml
	@echo "$(GREEN)environment.lock.yml genere.$(RESET)"

test: ## Lancer les tests unitaires (verifie que le code du validateur fonctionne)
	@echo "$(YELLOW)Lancement des tests unitaires...$(RESET)"
	@$(CONDA_RUN) python -m pytest tests/ -v
