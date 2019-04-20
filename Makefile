PROJECT = kmeans
VENV = venv

PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip


.PHONY: run install clean freeze

run:
	$(PY) $(PROJECT)/__main__.py

install:
	virtualenv -p python3 $(VENV)
	$(PIP) install -r requirements.txt
	$(PIP) install -r dev-requirements.txt

clean:
	rm -rf $(VENV)

freeze:
	$(PIP) freeze
