.PHONY: help
help: ## Show this help message
	@echo "Librarian Development Commands:\n"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Run the setup script to install uv and create environment
	@./setup.sh

.PHONY: install
install: ## Install the package in development mode
	@if ! command -v uv &> /dev/null; then \
		echo "Installing uv"; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	fi
	@uv pip install -e ".[dev]"

.PHONY: sync
sync: ## Sync dependencies from pyproject.toml
	@uv pip install -e ".[dev]"

.PHONY: build
build: clean-build ## Build wheel file
	@uv build

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@rm -rf dist build *.egg-info

.PHONY: clean
clean: clean-build ## Clean all generated files
	@rm -rf .pytest_cache .mypy_cache .coverage htmlcov .ruff_cache
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: test
test: ## Run tests with pytest
	@uv run pytest -v --cov --cov-config=pyproject.toml --cov-report=xml

.PHONY: test-fast
test-fast: ## Run tests without coverage
	@uv run pytest -v

.PHONY: coverage
coverage: ## Generate coverage report
	@uv run coverage report
	@uv run coverage html

.PHONY: lint
lint: ## Run linting with ruff
	@uv run ruff check librarian tests

.PHONY: lint-fix
lint-fix: ## Run linting with auto-fix
	@uv run ruff check --fix librarian tests

.PHONY: format
format: ## Format code with ruff
	@uv run ruff format librarian tests

.PHONY: format-check
format-check: ## Check code formatting
	@uv run ruff format --check librarian tests

.PHONY: typecheck
typecheck: ## Run type checking with mypy
	@uv run mypy librarian

.PHONY: check
check: lint format-check typecheck ## Run all code quality checks

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	@uv run pre-commit run -a

.PHONY: pre-commit-install
pre-commit-install: ## Install pre-commit hooks
	@uv run pre-commit install

.PHONY: evals
evals: ## Run Arcade tool evaluations
	@uv pip install -e ".[evals]"
	@uv run arcade evals . -p openai
