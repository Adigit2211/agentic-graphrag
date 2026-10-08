.PHONY: help install run test lint format typecheck ingest-sample docker-up docker-down

PY ?= python

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

install: ## Install the package with dev + embeddings extras
	$(PY) -m pip install -e ".[dev,embeddings]"

run: ## Start the API on http://localhost:8000 (needs an LLM backend, see README)
	uvicorn app.main:app --reload --port 8000

test: ## Run the full test suite (offline: LLM is mocked)
	pytest

lint: ## ruff + black --check
	ruff check .
	black --check .

format: ## Auto-format the code base
	black .
	ruff check --fix .

typecheck: ## mypy --strict
	mypy app ingestion graph retrieval agent llm scripts tests

ingest-sample: ## Upload data/sample/*.md to a running API
	$(PY) scripts/ingest_sample.py

docker-up: ## Build and start ollama + api (see README caveats)
	docker compose up --build

docker-down: ## Stop the compose stack
	docker compose down
