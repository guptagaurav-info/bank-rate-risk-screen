PY ?= python
export PYTHONPATH := src

.PHONY: all ingest run test
all: ingest run
ingest:
	$(PY) -m brs.ingest
run:
	$(PY) -m brs.pipeline
test:
	$(PY) -m pytest -q tests
