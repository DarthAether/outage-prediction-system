.PHONY: help install install-dev lint format typecheck test test-cov security \
       docker-up docker-down docker-build migrate pre-commit clean

PYTHON := python3
PIP := pip

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
install: ## Install production dependencies
	cd backend && $(PIP) install -e "."

install-dev: ## Install all dependencies (dev + test + ml)
	cd backend && $(PIP) install -e ".[ml,dev,test]"
	pre-commit install

# ─────────────────────────────────────────────────────────────────────────────
# Code Quality
# ─────────────────────────────────────────────────────────────────────────────
lint: ## Run ruff linter
	ruff check backend/src/ backend/tests/

format: ## Auto-format code with ruff
	ruff format backend/src/ backend/tests/
	ruff check --fix backend/src/ backend/tests/

typecheck: ## Run mypy type checking
	mypy backend/src/ --ignore-missing-imports

# ─────────────────────────────────────────────────────────────────────────────
# Testing
# ─────────────────────────────────────────────────────────────────────────────
test: ## Run unit tests
	cd backend && pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	cd backend && pytest tests/ \
		--cov=src \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=60 \
		-v

test-unit: ## Run only unit tests
	cd backend && pytest tests/unit/ -v

test-integration: ## Run only integration tests
	cd backend && pytest tests/integration/ -v

# ─────────────────────────────────────────────────────────────────────────────
# Security
# ─────────────────────────────────────────────────────────────────────────────
security: ## Run security scans (pip-audit + bandit)
	pip-audit
	bandit -r backend/src/ --severity-level medium

# ─────────────────────────────────────────────────────────────────────────────
# Database
# ─────────────────────────────────────────────────────────────────────────────
migrate: ## Run Alembic migrations
	cd backend && alembic upgrade head

migrate-new: ## Create new migration (usage: make migrate-new MSG="add xyz")
	cd backend && alembic revision --autogenerate -m "$(MSG)"

migrate-rollback: ## Rollback last migration
	cd backend && alembic downgrade -1

# ─────────────────────────────────────────────────────────────────────────────
# Docker
# ─────────────────────────────────────────────────────────────────────────────
docker-build: ## Build all Docker images
	docker compose -f docker/docker-compose.yml build

docker-up: ## Start all services
	docker compose -f docker/docker-compose.yml up -d

docker-down: ## Stop all services
	docker compose -f docker/docker-compose.yml down

docker-dev: ## Start in dev mode with hot-reload
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up

docker-logs: ## Tail logs from all services
	docker compose -f docker/docker-compose.yml logs -f

# ─────────────────────────────────────────────────────────────────────────────
# Pre-commit
# ─────────────────────────────────────────────────────────────────────────────
pre-commit: ## Run all pre-commit hooks
	pre-commit run --all-files

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────
clean: ## Remove build artifacts and caches
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov -exec rm -rf {} + 2>/dev/null || true
	rm -rf backend/.coverage backend/coverage.xml
