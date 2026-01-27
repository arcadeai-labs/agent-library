#!/bin/bash
# =============================================================================
# Librarian - UV Tool Setup
# =============================================================================
# Usage:
#   ./setup.sh              # Interactive mode (installs CLI globally)
#   ./setup.sh -y           # Non-interactive (auto-yes)
#   ./setup.sh --dev        # Development mode (also syncs dev dependencies)
#   ./setup.sh --help       # Show help
#
# This script uses `uv tool install -e .` to install the librarian CLI globally.
# No virtual environment activation needed - just run `librarian` or `libr`.
# =============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
PYTHON_VERSION="3.11"
AUTO_YES=false
DEV_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        --dev)
            DEV_MODE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-y|--yes] [--dev] [-h|--help]"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Non-interactive mode (auto-confirm all prompts)"
            echo "  --dev        Development mode (sync dev dependencies for testing/linting)"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "This script installs the librarian CLI globally using uv tool install."
            echo "After installation, 'librarian' and 'libr' commands are available everywhere."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Helper functions
confirm() {
    if [ "$AUTO_YES" = true ]; then
        return 0
    fi
    local prompt="$1"
    echo -ne "${YELLOW}${prompt} [Y/n]: ${NC}"
    read -r response
    response=${response:-y}
    [[ "$response" =~ ^[Yy] ]]
}

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║            Librarian - Context Management Service              ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Check/Install uv
echo -e "${YELLOW}[1/4] Checking uv installation...${NC}"
if ! command_exists uv; then
    echo -e "${RED}✗ uv is not installed${NC}"
    if confirm "Install uv now?"; then
        echo -e "  Installing uv..."
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
        echo -e "${GREEN}✓ uv installed${NC}"
    else
        echo -e "${RED}Cannot proceed without uv. Exiting.${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✓ uv is installed ($(uv --version 2>/dev/null | head -1))${NC}"
fi

# Step 2: Ensure Python is available
echo -e "\n${YELLOW}[2/4] Checking Python ${PYTHON_VERSION}...${NC}"
if ! uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then
    echo -e "  Python ${PYTHON_VERSION} not found, installing via uv..."
    uv python install "$PYTHON_VERSION"
fi
PYTHON_PATH=$(uv python find "$PYTHON_VERSION")
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} available at: ${PYTHON_PATH}${NC}"

# Step 3: Install librarian CLI globally using uv tool
echo -e "\n${YELLOW}[3/4] Installing librarian CLI globally...${NC}"

if [ -f "pyproject.toml" ]; then
    # Uninstall existing version if present (ignore errors)
    uv tool uninstall agent-library 2>/dev/null || true
    
    # Install as editable tool with specified Python version
    echo -e "  Installing with uv tool install -e . --python ${PYTHON_VERSION}..."
    uv tool install -e . --python "$PYTHON_VERSION"
    echo -e "${GREEN}✓ Librarian CLI installed globally${NC}"
    
    # Verify installation
    if command_exists librarian; then
        echo -e "${GREEN}✓ 'librarian' command is available${NC}"
    else
        echo -e "${YELLOW}  Note: You may need to add ~/.local/bin to your PATH${NC}"
        echo -e "${YELLOW}  Run: export PATH=\"\$HOME/.local/bin:\$PATH\"${NC}"
    fi
else
    echo -e "${RED}✗ No pyproject.toml found. Cannot install.${NC}"
    exit 1
fi

# Step 4: Setup directories and optional dev dependencies
echo -e "\n${YELLOW}[4/4] Setting up directories...${NC}"
mkdir -p ~/.librarian
mkdir -p documents
echo -e "${GREEN}✓ Directories created${NC}"

# Optional: Sync dev dependencies for development work
if [ "$DEV_MODE" = true ]; then
    echo -e "\n${YELLOW}[Dev] Syncing development dependencies...${NC}"
    uv sync --dev
    echo -e "${GREEN}✓ Dev dependencies synced (use 'uv run pytest' etc.)${NC}"
fi

# Done!
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete!                            ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}The CLI is now available globally (no venv activation needed):${NC}"
echo -e "  ${YELLOW}librarian --help${NC}"
echo -e "  ${YELLOW}libr --help${NC}"
echo ""
echo -e "${BLUE}Run the MCP server:${NC}"
echo -e "  ${YELLOW}librarian serve stdio${NC}    # For Claude Desktop"
echo -e "  ${YELLOW}librarian serve http${NC}     # For Cursor/VS Code"
echo ""
echo -e "${BLUE}For development (testing, linting, etc.):${NC}"
echo -e "  ${YELLOW}./setup.sh --dev${NC}         # Sync dev dependencies"
echo -e "  ${YELLOW}uv run pytest${NC}            # Run tests"
echo -e "  ${YELLOW}uv run ruff check .${NC}      # Run linting"
echo ""
echo -e "${BLUE}Configuration (via environment variables):${NC}"
echo -e "  ${YELLOW}DOCUMENTS_PATH${NC}    - Path to markdown files (default: ./documents)"
echo -e "  ${YELLOW}DATABASE_PATH${NC}     - SQLite database path (default: ~/.librarian/index.db)"
echo -e "  ${YELLOW}EMBEDDING_MODEL${NC}   - Sentence transformer model (default: all-MiniLM-L6-v2)"
echo -e "  ${YELLOW}CHUNK_SIZE${NC}        - Max chunk size in chars (default: 512)"
echo -e "  ${YELLOW}CHUNK_OVERLAP${NC}     - Overlap between chunks (default: 50)"
echo ""
if command_exists librarian; then
    echo -e "${BLUE}Verify installation:${NC}"
    echo -e "  $(which librarian)"
fi
