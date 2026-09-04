PYTHON ?= python
PYTEST ?= $(PYTHON) -m pytest

.PHONY: install test check run-api run-web

install:
	$(PYTHON) -m pip install -r requirements.txt

check:
	$(PYTHON) main.py check

test:
	$(PYTEST)

run-api:
	$(PYTHON) main.py serve-api

run-web:
	$(PYTHON) main.py serve-web
