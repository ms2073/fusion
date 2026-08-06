.DEFAULT_GOAL := help
.PHONY: help keys catalog up down restart logs stats test venv

help: ## Show this help
	@grep -E '^[a-z][a-z-]*:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

keys: ## Show which API keys are loaded (masked) and the resulting RPM ceiling
	@docker compose run --rm --no-deps --build fusion python -m fusion.keys

catalog: ## Check every configured model exists in the live NIM catalog
	@docker compose run --rm --no-deps --build fusion python -m fusion.catalog

probe: ## Measure which models your key can ACTUALLY call, and how fast
	@docker compose run --rm --no-deps --build fusion python -m fusion.probe

keytest: venv ## Determine whether NIM's queue is per-key or global (needs 2+ keys)
	@set -a && . ./.env && set +a && .venv/bin/python scripts/key_contention.py

up: ## Build and start the proxy
	@docker compose up -d --build
	@sleep 3 && curl -sf localhost:8787/healthz | python3 -m json.tool || \
		docker compose logs --tail 30

down: ## Stop the proxy
	@docker compose down

restart: ## Apply config.yaml changes (mounted, so no rebuild needed)
	@docker compose restart && echo "restarted"

logs: ## Follow the routing decisions
	@docker compose logs -f --no-log-prefix

stats: ## Show recent routing decisions and remaining headroom
	@curl -s localhost:8787/stats | python3 -m json.tool

test: venv ## Run the test suite
	@.venv/bin/python -m pytest -q

venv: ## Create the local dev environment
	@test -d .venv || (uv venv --python 3.12 .venv && \
		uv pip install --python .venv -q -e ".[dev]")
