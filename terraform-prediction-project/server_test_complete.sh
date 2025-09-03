#!/bin/bash
set -e

# Complete test script for Ubuntu server deployment
# Run this after deploying to the server

SERVER_HOST="manthram.tplinkdns.com"
SERVER_PORT="999" 
SERVER_USER="arnav"
REMOTE_PATH="/home/arnav/terraform-prediction-project"

echo "🧪 Complete Ubuntu Server Test Suite"
echo "===================================="

# Test 1: Basic connectivity and setup
echo "🔌 Test 1: Server Connectivity"
if ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'SSH connection working'" 2>/dev/null; then
    echo "✅ SSH connection successful"
else
    echo "❌ SSH connection failed"
    exit 1
fi

# Test 2: Check deployment
echo ""
echo "📦 Test 2: Deployment Check"
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST << 'EOF'
if [ -d "/home/arnav/terraform-prediction-project" ]; then
    echo "✅ Project directory exists"
    cd /home/arnav/terraform-prediction-project
    if [ -f "config/config.py" ]; then
        echo "✅ Project files deployed"
    else
        echo "❌ Project files missing"
        exit 1
    fi
else
    echo "❌ Project directory not found"
    exit 1
fi
EOF

# Test 3: Environment setup
echo ""
echo "🐍 Test 3: Environment Setup"
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST << 'EOF'
cd /home/arnav/terraform-prediction-project

if [ -d "terraform_venv" ]; then
    echo "✅ Virtual environment exists"
    source terraform_venv/bin/activate
    
    if python3.11 --version >/dev/null 2>&1; then
        echo "✅ Python 3.11 available"
    else
        echo "❌ Python 3.11 not found"
        exit 1
    fi
    
    if terraform version >/dev/null 2>&1; then
        echo "✅ Terraform CLI available"
    else
        echo "❌ Terraform CLI not found"
    fi
else
    echo "❌ Virtual environment not found"
    exit 1
fi
EOF

# Test 4: Dependencies
echo ""
echo "📦 Test 4: Python Dependencies"
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate

# Test critical imports
python -c "
try:
    import sys
    sys.path.insert(0, '.')
    from config.config import config
    print('✅ Configuration loaded')
except Exception as e:
    print(f'❌ Configuration failed: {e}')
    exit(1)

try:
    from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
    analyzer = TerraformAnalyzer()
    print('✅ Terraform analyzer loaded')
except Exception as e:
    print(f'❌ Terraform analyzer failed: {e}')
    exit(1)

try:
    from scripts.data_collection.github_collector import GitHubTerraformCollector
    print('✅ GitHub collector importable')
except Exception as e:
    print(f'❌ GitHub collector failed: {e}')
"
EOF

# Test 5: Environment variables
echo ""
echo "🔐 Test 5: Environment Variables"
echo "⚠️  Make sure to set GITHUB_TOKEN on the server"

# Test 6: Basic functionality
echo ""
echo "⚙️  Test 6: Basic Functionality" 
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate

# Set PYTHONPATH
export PYTHONPATH=/home/arnav/terraform-prediction-project

# Load environment if available
if [ -f ".env" ]; then
    source .env
fi

# Test basic functionality
python -c "
import sys
sys.path.insert(0, '.')
from scripts.data_collection.terraform_analyzer import TerraformAnalyzer

analyzer = TerraformAnalyzer()

# Test analysis
test_tf = '''
resource \"aws_s3_bucket\" \"test\" {
  bucket = \"test-bucket\"
}
'''

result = analyzer.analyze_terraform_content(test_tf)
if result and len(result['resources']) > 0:
    print('✅ Terraform analysis working')
    print(f'   Found {len(result[\"resources\"])} resources')
else:
    print('❌ Terraform analysis failed')
    exit(1)
"
EOF

# Test 7: Data collection test
echo ""
echo "📊 Test 7: Minimal Data Collection (Optional - requires GITHUB_TOKEN)"
echo "To run this test, first set GITHUB_TOKEN on the server:"
echo "ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
echo "cd $REMOTE_PATH && source terraform_venv/bin/activate"
echo "export GITHUB_TOKEN='your_token_here'"
echo "python scripts/pipeline.py --phase data_collection --max-repos 1"

# Test 8: API server test
echo ""
echo "🌐 Test 8: API Server"
ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
export PYTHONPATH=/home/arnav/terraform-prediction-project

# Start API server in background
python server/api.py --host 127.0.0.1 --port 8001 > api_test.log 2>&1 &
API_PID=$!

# Wait for startup
sleep 5

# Test API
if curl -s http://127.0.0.1:8001/health >/dev/null; then
    echo "✅ API server responding"
    curl -s http://127.0.0.1:8001/health | python -m json.tool
else
    echo "❌ API server not responding"
    cat api_test.log
fi

# Cleanup
kill $API_PID 2>/dev/null || true
rm -f api_test.log
EOF

echo ""
echo "🎉 Server Test Suite Completed!"
echo ""
echo "Next Steps:"
echo "1. Set environment variables on server:"
echo "   ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
echo "   nano $REMOTE_PATH/.env"
echo ""
echo "2. Run basic test:"
echo "   ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST 'cd $REMOTE_PATH && source terraform_venv/bin/activate && source .env && python demo_basic.py'"
echo ""
echo "3. Test data collection:"
echo "   ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST 'cd $REMOTE_PATH && source terraform_venv/bin/activate && source .env && python scripts/pipeline.py --phase data_collection --max-repos 2'"
echo ""
echo "4. Start API server:"
echo "   ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST 'cd $REMOTE_PATH && source terraform_venv/bin/activate && source .env && screen -S api -d -m python server/api.py --host 0.0.0.0 --port 8000'"
echo ""
echo "5. Test from your Mac:"
echo "   curl http://$SERVER_HOST:8000/health"
