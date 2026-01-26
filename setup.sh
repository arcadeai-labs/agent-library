#!/bin/bash
# =============================================================================
# Librarian - UV Environment Setup
# =============================================================================
# Usage:
#   ./setup.sh              # Interactive mode
#   ./setup.sh -y           # Non-interactive (auto-yes)
#   ./setup.sh --help       # Show help
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
VENV_DIR=".venv"
AUTO_YES=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [-y|--yes] [-h|--help]"
            echo ""
            echo "Options:"
            echo "  -y, --yes    Non-interactive mode (auto-confirm all prompts)"
            echo "  -h, --help   Show this help message"
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
echo -e "${YELLOW}[1/5] Checking uv installation...${NC}"
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
echo -e "\n${YELLOW}[2/5] Checking Python ${PYTHON_VERSION}...${NC}"
if ! uv python find "$PYTHON_VERSION" >/dev/null 2>&1; then
    echo -e "  Python ${PYTHON_VERSION} not found, installing via uv..."
    uv python install "$PYTHON_VERSION"
fi
PYTHON_PATH=$(uv python find "$PYTHON_VERSION")
echo -e "${GREEN}✓ Python ${PYTHON_VERSION} available at: ${PYTHON_PATH}${NC}"

# Step 3: Create virtual environment
echo -e "\n${YELLOW}[3/5] Setting up virtual environment...${NC}"
if [ -d "$VENV_DIR" ]; then
    CURRENT_VERSION=$("$VENV_DIR/bin/python" --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
    if [ "$CURRENT_VERSION" = "$PYTHON_VERSION" ]; then
        echo -e "${GREEN}✓ Virtual environment exists with Python ${CURRENT_VERSION}${NC}"
    else
        echo -e "  Recreating venv (current: Python ${CURRENT_VERSION})..."
        rm -rf "$VENV_DIR"
        uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
        echo -e "${GREEN}✓ Virtual environment created with Python ${PYTHON_VERSION}${NC}"
    fi
else
    uv venv --python "$PYTHON_VERSION" "$VENV_DIR"
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi

# Step 4: Install dependencies from pyproject.toml
echo -e "\n${YELLOW}[4/5] Installing dependencies...${NC}"

if [ -f "pyproject.toml" ]; then
    echo -e "  Installing package in development mode with dev dependencies..."
    uv pip install -e ".[dev]"
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ No pyproject.toml found. Cannot install dependencies.${NC}"
    exit 1
fi

# Step 5: Setup directories and environment
echo -e "\n${YELLOW}[5/5] Setting up directories...${NC}"
mkdir -p ~/.librarian
mkdir -p documents
echo -e "${GREEN}✓ Directories created${NC}"

# Done!
echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete! ✓                          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Activate the environment:${NC}"
echo -e "  ${YELLOW}source ${VENV_DIR}/bin/activate${NC}"
echo ""
echo -e "${BLUE}Run the MCP server:${NC}"
echo -e "  ${YELLOW}uv run librarian/server.py stdio${NC}    # For Claude Desktop"
echo -e "  ${YELLOW}uv run librarian/server.py http${NC}     # For Cursor/VS Code"
echo ""
echo -e "${BLUE}Configuration (via environment variables):${NC}"
echo -e "  ${YELLOW}DOCUMENTS_PATH${NC}    - Path to markdown files (default: ./documents)"
echo -e "  ${YELLOW}DATABASE_PATH${NC}     - SQLite database path (default: ~/.librarian/index.db)"
echo -e "  ${YELLOW}EMBEDDING_MODEL${NC}   - Sentence transformer model (default: all-MiniLM-L6-v2)"
echo -e "  ${YELLOW}CHUNK_SIZE${NC}        - Max chunk size in chars (default: 512)"
echo -e "  ${YELLOW}CHUNK_OVERLAP${NC}     - Overlap between chunks (default: 50)"
echo ""
echo -e "${BLUE}Environment info:${NC}"
echo -e "  Python: ${VENV_DIR}/bin/python ($(${VENV_DIR}/bin/python --version 2>&1))"
echo -e "  Packages: $(${VENV_DIR}/bin/pip list 2>/dev/null | wc -l | tr -d ' ') installed"
