.PHONY: help requirements install format lint test coverage


SHELL := /bin/bash
PLATFORM := $(shell \
  if [[ -n $$(command -v nvidia-smi && nvidia-smi --list-gpus) ]]; then echo gpu; \
  elif [[ $$(uname -s) == Darwin ]]; then echo osx; \
  else echo cpu; \
  fi)

PIP_COMPILE_ARGS ?=

help:
	@echo "Available commands:"
	@echo "requirements       compile all requirements."
	@echo "install            install dev requirements for CPU only."
	@echo "format             format code."
	@echo "lint               run linters."
	@echo "test               run unit tests."
	@echo "coverage           build coverage report."

requirements:
	pip install -q --upgrade pip-tools
	pip-compile -q ${PIP_COMPILE_ARGS} requirements/requirements.in \
	  --extra-index-url https://download.pytorch.org/whl/cpu \
	  --output-file requirements/requirements-cpu.txt \
	  --strip-extras \
	  --resolver=backtracking
	pip-compile -q ${PIP_COMPILE_ARGS} requirements/requirements.in \
	  --extra-index-url https://download.pytorch.org/whl/cu117 \
	  --output-file requirements/requirements-gpu.txt \
	  --strip-extras \
	  --resolver=backtracking
	pip-compile -q ${PIP_COMPILE_ARGS} requirements/requirements.in \
	  --output-file requirements/requirements-osx.txt \
	  --strip-extras \
	  --resolver=backtracking
	pip-compile -q ${PIP_COMPILE_ARGS} requirements/requirements-ci.in \
	  --strip-extras \
	  --resolver=backtracking
	pip-compile -q ${PIP_COMPILE_ARGS} requirements/requirements-dev.in \
	  --strip-extras \
	  --resolver=backtracking


.venv_timestamp: requirements/*.txt
	pip install -q --upgrade pip-tools
ifdef GITLAB_CI
	pip-sync -q \
	  requirements/requirements-cpu.txt \
	  requirements/requirements-ci.txt
else
	pip-sync -q \
	  requirements/requirements-${PLATFORM}.txt \
	  requirements/requirements-ci.txt \
	  requirements/requirements-dev.txt
endif
ifneq ($(wildcard requirements/requirements-extra.txt),)
	pip install -q -r requirements/requirements-extra.txt
endif
	pip install -q -e .
	touch .venv_timestamp

install: .venv_timestamp

# format: install
# 	isort dirpa tests
# 	black dirpa tests

# mypy: install
# 	mypy dirpa tests

# flake8: install
# 	flake8 dirpa tests

lint: mypy flake8

test: install
	pytest -v

integration-test: install
	pytest -v -m integration

coverage: install
	coverage run --source dirpa -m pytest -v --junit-xml=report.xml
	coverage report
	coverage xml
