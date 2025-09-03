#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Server configuration (from .env.template)
SERVER_HOST="manthram.tplinkdns.com"
SERVER_PORT="999"
SERVER_USER="arnav"
REMOTE_PATH="/home/arnav/terraform-prediction-project"

echo -e "${BLUE}🚀 Terraform Prediction Model - Server Deployment${NC}"
echo "================================================================="

# Check if we're in the correct directory
if [ ! -f "config/config.py" ]; then
    echo -e "${RED}❌ Not in project directory. Please run from terraform-prediction-project/${NC}"
    exit 1
fi

echo -e "${YELLOW}📁 Current directory: $(pwd)${NC}"
echo -e "${YELLOW}🖥️  Target server: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}${NC}"

# Test SSH connection
echo -e "${BLUE}🔌 Testing SSH connection...${NC}"
if ! ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "echo 'Connection test successful'" 2>/dev/null; then
    echo -e "${RED}❌ Cannot connect to server. Check your SSH configuration.${NC}"
    echo "Try: ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
    exit 1
fi
echo -e "${GREEN}✅ SSH connection working${NC}"

# Create deployment package
echo -e "${BLUE}📦 Creating deployment package...${NC}"
tar -czf terraform-prediction-deploy.tar.gz \
    --exclude='terraform_venv' \
    --exclude='cache' \
    --exclude='data' \
    --exclude='models' \
    --exclude='logs' \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    .

echo -e "${GREEN}✅ Package created: $(du -h terraform-prediction-deploy.tar.gz | cut -f1)${NC}"

# Transfer to server
echo -e "${BLUE}📤 Transferring to server...${NC}"
scp -P ${SERVER_PORT} terraform-prediction-deploy.tar.gz ${SERVER_USER}@${SERVER_HOST}:~/

# Deploy on server
echo -e "${BLUE}🚀 Deploying on server...${NC}"
ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
set -e

echo "🐧 Setting up on Ubuntu server..."

# Create project directory
mkdir -p /home/arnav/terraform-prediction-project
cd /home/arnav/terraform-prediction-project

# Extract new deployment
tar -xzf ~/terraform-prediction-deploy.tar.gz

echo "📦 Project extracted"

# Update system (if needed)
sudo apt update

# Install system dependencies if not present
if ! command -v python3.11 &> /dev/null; then
    echo "Installing Python 3.11..."
    sudo apt install -y python3.11 python3.11-venv python3.11-dev
fi

if ! command -v terraform &> /dev/null; then
    echo "Installing Terraform..."
    wget -O terraform.zip https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
    unzip terraform.zip
    sudo mv terraform /usr/local/bin/
    sudo chmod +x /usr/local/bin/terraform
    rm terraform.zip
fi

# Install other dependencies
sudo apt install -y git wget unzip curl jq htop build-essential

# Create virtual environment
if [ ! -d "terraform_venv" ]; then
    echo "Creating virtual environment..."
    python3.11 -m venv terraform_venv
fi

source terraform_venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install Python dependencies
echo "Installing Python packages (this may take 10-15 minutes)..."
pip install -r requirements.txt

# Create directories
mkdir -p data/{raw,processed,ground_truth}
mkdir -p models/{checkpoints,fine_tuned}
mkdir -p logs cache tests

# Set up environment
if [ ! -f ".env" ]; then
    cp .env.template .env
    echo "⚠️  Please edit .env file with your tokens:"
    echo "   nano .env"
fi

echo "✅ Server deployment completed!"
echo ""
echo "Next steps:"
echo "1. Edit .env file: nano .env"
echo "2. Load environment: source .env"
echo "3. Test setup: python test_setup.py"
echo "4. Run pipeline: python scripts/pipeline.py --phase data_collection --max-repos 5"

ENDSSH

# Clean up local deployment package
rm terraform-prediction-deploy.tar.gz

echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}Next steps on the server:${NC}"
echo "1. SSH to server: ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
echo "2. Navigate to project: cd ${REMOTE_PATH}"
echo "3. Edit environment: nano .env"
echo "4. Test setup: python test_setup.py"
echo ""
echo -e "${BLUE}To run the project:${NC}"
echo "ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} 'cd ${REMOTE_PATH} && source terraform_venv/bin/activate && source .env && python test_setup.py'"
