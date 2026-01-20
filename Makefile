.PHONY: help
help: ## Show this help message
	@echo "🛠️  Librarian Development Commands:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Run the setup script to install uv and create environment
	@./setup.sh

.PHONY: install
install: ## Install the package in development mode with all dependencies
	@echo "📦 Checking if uv is installed"
	@if ! command -v uv &> /dev/null; then \
		echo "📦 Installing uv"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo "📦 uv is already installed"; \
	fi
	@echo "🚀 Installing package in development mode with dev dependencies"
	@uv pip install -e ".[dev]"

.PHONY: sync
sync: ## Sync dependencies from pyproject.toml
	@echo "🔄 Syncing dependencies"
	@uv pip install -e ".[dev]"

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uv build

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@rm -rf dist build *.egg-info

.PHONY: clean
clean: clean-build ## Clean all generated files
	@echo "🗑️  Cleaning generated files"
	@rm -rf .pytest_cache .mypy_cache .coverage htmlcov .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: clean-index
clean-index: ## Clean the index database
	@echo "🗑️  Removing index database"
	@rm -rf ~/.librarian/index.db

.PHONY: test
test: ## Run tests with pytest
	@echo "🧪 Running tests"
	@uv run pytest -W ignore -v --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: test-fast
test-fast: ## Run tests without coverage
	@echo "🧪 Running tests (fast mode)"
	@uv run pytest -W ignore -v

.PHONY: coverage
coverage: ## Generate coverage report
	@echo "📊 Generating coverage report"
	@uv run coverage report
	@uv run coverage html
	@echo "📊 Coverage report generated at htmlcov/index.html"

.PHONY: lint
lint: ## Run linting with ruff
	@echo "🔍 Linting code with ruff"
	@uv run ruff check librarian tests

.PHONY: lint-fix
lint-fix: ## Run linting with auto-fix
	@echo "🔧 Fixing lint issues"
	@uv run ruff check --fix librarian tests

.PHONY: format
format: ## Format code with ruff
	@echo "✨ Formatting code"
	@uv run ruff format librarian tests

.PHONY: format-check
format-check: ## Check code formatting
	@echo "🔍 Checking code format"
	@uv run ruff format --check librarian tests

.PHONY: typecheck
typecheck: ## Run type checking with mypy
	@echo "🔍 Running mypy type checks"
	@uv run mypy librarian

.PHONY: check
check: lint format-check typecheck ## Run all code quality checks
	@echo "✅ All checks passed"

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	@echo "🪝 Running pre-commit hooks"
	@uv run pre-commit run -a

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	@echo "🪝 Installing pre-commit hooks"
	@uv run pre-commit install

.PHONY: run-stdio
run-stdio: ## Run the MCP server with stdio transport
	@echo "🚀 Starting MCP server (stdio)"
	@uv run librarian/server.py stdio

.PHONY: run-http
run-http: ## Run the MCP server with HTTP transport
	@echo "🚀 Starting MCP server (HTTP on port 8000)"
	@uv run librarian/server.py http

.PHONY: ingest
ingest: ## Ingest documents from the default documents directory
	@echo "📥 Ingesting documents from ./documents"
	@uv run python -c "import asyncio; from librarian.server import ingest_directory; print(asyncio.run(ingest_directory(None, '', True, False)))"

.PHONY: stats
stats: ## Show index statistics
	@echo "📊 Index statistics"
	@uv run python -c "from librarian.storage.database import get_database; import json; print(json.dumps(get_database().get_stats(), indent=2))"

.PHONY: evals
evals: ## Run Arcade tool evaluations
	@echo "🧪 Running Arcade evaluations"
	@uv pip install -e ".[evals]"
	@uv run arcade evals . -p openai
