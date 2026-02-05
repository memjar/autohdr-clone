#!/bin/bash
#
# AutoHDR Clone - Backend Startup Script
# ======================================
# Starts the Python backend with full RAW file support
#
# Usage:
#   ./start-backend.sh          # Normal start
#   ./start-backend.sh --setup  # First-time setup + start
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}     AutoHDR Clone - Python Backend                           ${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# ============================================
# STEP 1: Check Python
# ============================================
echo -e "${YELLOW}[1/5]${NC} Checking Python..."

if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}ERROR: Python not found!${NC}"
    echo "Install Python 3.9+ from https://python.org"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "  ✓ Found Python ${GREEN}$PYTHON_VERSION${NC}"

# ============================================
# STEP 2: Virtual Environment
# ============================================
echo -e "${YELLOW}[2/5]${NC} Checking virtual environment..."

VENV_DIR="$SCRIPT_DIR/.venv"

if [[ "$1" == "--setup" ]] || [[ ! -d "$VENV_DIR" ]]; then
    echo -e "  Creating virtual environment..."
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo -e "  ✓ Created ${GREEN}.venv${NC}"
else
    echo -e "  ✓ Found existing ${GREEN}.venv${NC}"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# ============================================
# STEP 3: Install Dependencies
# ============================================
echo -e "${YELLOW}[3/5]${NC} Checking dependencies..."

# Check if we need to install
NEED_INSTALL=false
if [[ "$1" == "--setup" ]]; then
    NEED_INSTALL=true
elif ! $PYTHON_CMD -c "import cv2, rawpy, fastapi, uvicorn" 2>/dev/null; then
    NEED_INSTALL=true
fi

if $NEED_INSTALL; then
    echo -e "  Installing dependencies (this may take a minute)..."
    pip install --upgrade pip -q
    pip install -r backend-requirements.txt -q
    echo -e "  ✓ Dependencies installed"
else
    echo -e "  ✓ All dependencies present"
fi

# ============================================
# STEP 4: Verify RAW Support
# ============================================
echo -e "${YELLOW}[4/5]${NC} Verifying RAW file support..."

$PYTHON_CMD -c "
import rawpy
print('  ✓ rawpy loaded - ARW/CR2/NEF support enabled')
" 2>/dev/null || {
    echo -e "${RED}  ✗ rawpy failed to load${NC}"
    echo -e "  Try: pip install rawpy"
}

$PYTHON_CMD -c "
import cv2
print(f'  ✓ OpenCV {cv2.__version__} loaded')
" 2>/dev/null || {
    echo -e "${RED}  ✗ OpenCV failed to load${NC}"
    echo -e "  Try: pip install opencv-python"
}

# ============================================
# STEP 5: Start Server
# ============================================
echo -e "${YELLOW}[5/5]${NC} Starting backend server..."
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Backend running at: http://localhost:8000                    ${NC}"
echo -e "${GREEN}  API docs at:        http://localhost:8000/docs               ${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${YELLOW}In your browser:${NC}"
echo -e "  1. Go to https://autohdr-clone.vercel.app"
echo -e "  2. Click the toggle to switch to '${GREEN}Local Backend${NC}'"
echo -e "  3. Upload your ARW files and process!"
echo ""
echo -e "  Press ${RED}Ctrl+C${NC} to stop the server"
echo ""

# Run uvicorn
$PYTHON_CMD -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
