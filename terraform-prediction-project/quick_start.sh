#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Terraform Prediction Model - Quick Start${NC}"
echo "=================================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

echo -e "${YELLOW}📁 Current directory: $(pwd)${NC}"

# Check if Python 3.11+ is available
echo -e "${BLUE}🐍 Checking Python version...${NC}"
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    if [[ "$(echo "$PYTHON_VERSION >= 3.11" | bc)" -eq 1 ]]; then
        PYTHON_CMD="python3"
    else
        echo -e "${RED}❌ Python 3.11+ required, found $PYTHON_VERSION${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ Python 3.11+ not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Using $PYTHON_CMD${NC}"

# Check if virtual environment exists
if [ ! -d "terraform_venv" ]; then
    echo -e "${BLUE}🔧 Creating virtual environment...${NC}"
    $PYTHON_CMD -m venv terraform_venv
fi

# Activate virtual environment
echo -e "${BLUE}🔌 Activating virtual environment...${NC}"
source terraform_venv/bin/activate

# Upgrade pip
echo -e "${BLUE}⬆️  Upgrading pip...${NC}"
pip install --upgrade pip setuptools wheel

# Install requirements
echo -e "${BLUE}📦 Installing Python dependencies...${NC}"
pip install -r requirements.txt

# Create necessary directories
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p data/{raw,processed,ground_truth}
mkdir -p models/{checkpoints,fine_tuned}
mkdir -p logs cache tests

# Check if Terraform is installed
echo -e "${BLUE}🔧 Checking Terraform installation...${NC}"
if command -v terraform &> /dev/null; then
    TERRAFORM_VERSION=$(terraform version | head -n 1)
    echo -e "${GREEN}✅ $TERRAFORM_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  Terraform not found. Installing via Homebrew...${NC}"
    if command -v brew &> /dev/null; then
        brew install terraform
        echo -e "${GREEN}✅ Terraform installed${NC}"
    else
        echo -e "${RED}❌ Homebrew not found. Please install Terraform manually:${NC}"
        echo "   brew install terraform"
        exit 1
    fi
fi

# Check environment variables
echo -e "${BLUE}🔐 Checking environment variables...${NC}"
if [ -f ".env" ]; then
    echo -e "${GREEN}✅ .env file exists${NC}"
    source .env
else
    echo -e "${YELLOW}⚠️  Creating .env file from template...${NC}"
    cp .env.template .env
    echo -e "${YELLOW}📝 Please edit .env file with your tokens:${NC}"
    echo "   nano .env"
    echo "   # Add your GITHUB_TOKEN and other credentials"
fi

# Test basic functionality
echo -e "${BLUE}🧪 Running setup tests...${NC}"
$PYTHON_CMD test_setup.py

echo ""
echo -e "${GREEN}🎉 Quick start completed!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Set your environment variables in .env file"
echo "2. Run: source .env"
echo "3. Test with: python examples/run_example.py --mode pipeline"
echo "4. Or run: make pipeline-small"
echo ""
echo -e "${BLUE}To activate the environment in future sessions:${NC}"
echo "   cd $(pwd)"
echo "   source terraform_venv/bin/activate"
echo "   source .env"
echo ""
echo -e "${BLUE}For full documentation, see:${NC}"
echo "   📖 README.md"
echo "   🛠️  SETUP_GUIDE.md"
