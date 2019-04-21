PROJECT = kmeans
VENV = venv
BIN = $(VENV)/bin

PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

MAIN := $(PROJECT)/__main__.py


.PHONY: run install clean freeze

DATASET := iris

run:
	$(PY) $(MAIN) $(DATASET)

help:
	$(PY) $(MAIN) -h

install:
	virtualenv -p python3 $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(BIN)/pre-commit install

clean:
	rm -rf $(VENV)

freeze:
	$(PIP) freeze
