#################################################################################
# GLOBALS
#################################################################################

PROJECT_NAME = collision-severity-predictor
PYTHON_VERSION = 3.12

#################################################################################
# COMMANDS
#################################################################################

## Install dependencies (Poetry)
.PHONY: pipeline
pipeline:
	poetry run python -m src.pipeline

.PHONY: install
install:
	poetry install


## Run feature engineering
.PHONY: features
features:
	poetry run python -m src.features.build_features


## Run model / experiments
.PHONY: train
train:
	poetry run python -m src.modeling.train

.PHONY: eval
eval:
	poetry run python -m src.modeling.eval


## Run tests with coverage
.PHONY: test
test:
	poetry run pytest tests --cov=src --cov-report=term-missing


## Lint code
.PHONY: lint
lint:
	poetry run ruff check .


## Format code
.PHONY: format
format:
	poetry run ruff check --fix .
	poetry run ruff format .


## Clean cache
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# CI PIPELINE (IMPORTANT)
#################################################################################

## Full pipeline (what CI will run)
.PHONY: ci
ci: install lint test


#################################################################################
# HELP
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@python -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)