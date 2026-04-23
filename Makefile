.EXPORT_ALL_VARIABLES:
SHELL = bash

CONDA_ENV ?= gmic
CONDA_RUN := conda run -n $(CONDA_ENV) --no-capture-output
NOTEBOOK  ?= gmic

.DEFAULT_GOAL := help
.PHONY: help build run test notebook freeze

help:  ## Affiche cette aide
	@grep -E '^[a-zA-Z_-]+:.*?##' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

build:  ## Cree l'environnement conda gmic depuis environment.yml
	conda env create -f environment.yml || conda env update -f environment.yml --prune

run:  ## Sert un notebook Quarto en live preview (NOTEBOOK=gmic|resnet18_training|rsna_comparison)
	$(CONDA_RUN) quarto preview script_notebook/$(NOTEBOOK).qmd

notebook:  ## Rend un notebook Quarto en HTML (NOTEBOOK=gmic|resnet18_training|rsna_comparison)
	$(CONDA_RUN) quarto render script_notebook/$(NOTEBOOK).qmd --to html

test:  ## Lance les tests unitaires
	$(CONDA_RUN) pytest tests/

freeze:  ## Exporte les versions exactes (environment.lock.yml)
	conda env export -n $(CONDA_ENV) --no-builds > environment.lock.yml
