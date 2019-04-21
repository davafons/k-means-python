PROJECT = kmeans
VENV = venv
BIN = $(VENV)/bin

PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip


.PHONY: run install clean freeze

run:
	$(PY) $(PROJECT)/__main__.py

install:
	virtualenv -p python3 $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-dev.txt
	$(BIN)/pre-commit install

clean:
	rm -rf $(VENV)

freeze:
	$(PIP) freeze
