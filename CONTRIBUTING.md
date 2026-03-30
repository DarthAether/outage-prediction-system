# Contributing to Outage Prediction System

## Development Setup

### Prerequisites

- Python 3.11+
- Node.js 20+
- Docker & Docker Compose
- Make

### Quick Start

```bash
# Clone the repository
git clone https://github.com/DarthAether/outage-prediction-system.git
cd outage-prediction-system

# Install backend dependencies (with dev + test + ML extras)
make install-dev

# Copy environment config
cp .env.example .env
# Edit .env with your API keys and a secure DB_PASSWORD

# Start infrastructure (DB, Redis, MLflow)
make docker-up

# Run tests
make test

# Run linter
make lint
```

### Code Quality

This project enforces code quality via pre-commit hooks and CI:

| Tool | Purpose | Command |
|------|---------|---------|
| **Ruff** | Linting + formatting | `make lint` / `make format` |
| **Mypy** | Type checking | `make typecheck` |
| **Pytest** | Unit + integration tests | `make test` |
| **pip-audit** | Dependency vulnerabilities | `make security` |
| **Bandit** | Source code security scan | `make security` |

Pre-commit hooks run automatically on `git commit`. To run manually:

```bash
make pre-commit
```

### Branch Strategy

- `main` — production-ready, protected
- `develop` — integration branch
- Feature branches: `feature/<description>`
- Bug fixes: `fix/<description>`

### Pull Request Process

1. Create a feature branch from `develop`
2. Make changes, ensuring all checks pass (`make lint test typecheck`)
3. Push and open a PR against `develop`
4. Fill out the PR template
5. Wait for CI to pass and request review

### Testing

```bash
make test          # All tests
make test-unit     # Unit tests only
make test-cov      # With coverage report (60% minimum)
```

### Docker Development

```bash
make docker-dev    # Hot-reload mode for all services
make docker-logs   # Tail logs
make docker-down   # Stop everything
```

### Database Migrations

```bash
make migrate                        # Apply pending migrations
make migrate-new MSG="add column"   # Generate a new migration
make migrate-rollback               # Roll back last migration
```
