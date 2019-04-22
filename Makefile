PROJECT = kmeans
VENV = venv
BIN = $(VENV)/bin

PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip


.PHONY: run help install clean freeze

DATASET := iris

run:
	$(PY) -m kmeans $(DATASET)

help:
	$(PY) -m kmeans $(MAIN) -h

install:
	virtualenv -p python3 $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(BIN)/pre-commit install

clean:
	rm -rf $(VENV)

freeze:
	$(PIP) freeze
