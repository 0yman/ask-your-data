.PHONY: help install warehouse serve ask test lint eval eval-quick docker clean

help:
	@echo "install      Install runtime + dev dependencies"
	@echo "warehouse    Build data/port.duckdb from a fixed seed"
	@echo "serve        Run the API on http://localhost:8000"
	@echo "ask Q='...'  Ask one question from the command line"
	@echo "test         Run the test suite (offline, no API key)"
	@echo "lint         Run ruff"
	@echo "eval         Run the full evaluation (needs GOOGLE_API_KEY)"
	@echo "eval-quick   Evaluate the first 5 questions only"
	@echo "docker       Build and run the container"

install:
	pip install -r requirements-dev.txt

warehouse:
	python scripts/build_warehouse.py

serve:
	uvicorn agent.api:app --app-dir src --reload --port 8000

ask:
	@PYTHONPATH=src python -m agent.cli --trace "$(Q)"

test:
	python -m pytest

lint:
	ruff check src eval scripts tests

eval:
	python eval/run_eval.py

eval-quick:
	python eval/run_eval.py --limit 5

docker:
	docker compose up --build

clean:
	rm -rf data/*.duckdb .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
