# 🐧 How to Run on Ubuntu Server (manthram.tplinkdns.com)

Complete guide for deploying and running the Terraform prediction model on your Ubuntu server.

## 🚀 Quick Start (5 commands)

```bash
# 1. Deploy from your Mac
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
./deploy_to_server.sh

# 2. SSH to your server
ssh -p 999 arnav@manthram.tplinkdns.com

# 3. Setup environment
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
nano .env  # Add your GITHUB_TOKEN

# 4. Test setup
source .env && python test_setup.py

# 5. Run demo
python demo_basic.py
```

## 📋 Complete Step-by-Step Instructions

### Step 1: Deploy to Server

From your Mac terminal:

```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# Deploy automatically
./deploy_to_server.sh

# Or test the connection first
ssh -p 999 arnav@manthram.tplinkdns.com "echo 'Server connection working'"
```

### Step 2: Complete Server Setup

SSH to your server:

```bash
ssh -p 999 arnav@manthram.tplinkdns.com
```

On the server:

```bash
# Navigate to project
cd /home/arnav/terraform-prediction-project

# Activate environment
source terraform_venv/bin/activate

# Verify system dependencies
python3.11 --version  # Should be 3.11+
terraform version      # Should show Terraform installed
git --version         # Should show Git installed

# Edit environment variables
nano .env

# Add these required variables to .env:
# GITHUB_TOKEN=ghp_your_actual_github_personal_access_token
# HF_TOKEN=hf_your_huggingface_token_if_you_have_one

# Load environment
source .env

# Set Python path
export PYTHONPATH=/home/arnav/terraform-prediction-project
echo 'export PYTHONPATH=/home/arnav/terraform-prediction-project' >> ~/.bashrc
```

### Step 3: Test Basic Setup

```bash
# Still on the server
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Run setup test
python test_setup.py

# Run basic demo
python demo_basic.py

# Test individual components
python -c "
from config.config import config
print(f'✅ Model: {config.model_name}')

from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
analyzer = TerraformAnalyzer()

test_tf = '''resource \"aws_s3_bucket\" \"test\" { bucket = \"test\" }'''
result = analyzer.analyze_terraform_content(test_tf)
print(f'✅ Analysis: Found {len(result[\"resources\"])} resources')
"
```

## 🧪 Testing Data Collection

### Quick Test (5 minutes)

```bash
# On server
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Test with 1 repository
python scripts/pipeline.py --phase data_collection --max-repos 1

# Check what was created
ls -la data/ground_truth/
```

### Medium Test (30-60 minutes)

```bash
# Run in screen to prevent disconnection
screen -S data-collection

# Inside screen
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Collect from multiple repositories
python scripts/pipeline.py --phase data_collection --max-repos 10

# Monitor progress in another terminal:
# ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/pipeline.log'

# Detach from screen: Ctrl+A, then D
```

### Process the Data

```bash
# After data collection completes
python scripts/pipeline.py --phase data_processing

# Check results
ls -la data/processed/
cat data/processed/dataset_info.json | jq .
```

## 🤖 Model Training on Server

### Check Server Resources

```bash
# Check available resources
free -h                    # Available RAM
nproc                     # CPU cores  
nvidia-smi               # GPU (if available)
df -h /home/arnav        # Disk space
```

### Training Test (30-60 minutes)

```bash
# Start in screen session
screen -S training

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Test model loading first
python -c "
from scripts.model_training.model_setup import TerraformModelSetup
print('Loading model...')
setup = TerraformModelSetup() 
model, tokenizer = setup.setup_model_and_tokenizer()
print('✅ Model loaded successfully')
"

# Quick training test
python scripts/pipeline.py --phase training --max-iterations 1

# Detach: Ctrl+A, D
# Monitor: ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/training.log'
```

### Full Pipeline (2-4 hours)

```bash
# Run complete pipeline
screen -S full-pipeline

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Full training pipeline
python scripts/pipeline.py --phase full \
  --max-repos 25 \
  --max-iterations 2 \
  --performance-threshold 0.7

# Detach and monitor progress remotely
```

## 🌐 API Server Deployment

### Start API Server

```bash
# On server
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Option 1: Interactive (for testing)
python server/api.py --host 0.0.0.0 --port 8000

# Option 2: Background process
nohup python server/api.py --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &

# Option 3: Screen session (recommended)
screen -S terraform-api
python server/api.py --host 0.0.0.0 --port 8000
# Ctrl+A, D to detach

# Option 4: Systemd service (production)
sudo systemctl start terraform-api  # If service is configured
```

### Test API Server

```bash
# From your Mac
curl http://manthram.tplinkdns.com:8000/health

# Test prediction endpoint
curl -X POST "http://manthram.tplinkdns.com:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-test-bucket\"\n}",
    "max_tokens": 256
  }' | jq .

# Test with Azure example
curl -X POST "http://manthram.tplinkdns.com:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"azurerm_resource_group\" \"example\" {\n  name = \"test-rg\"\n  location = \"East US\"\n}",
    "max_tokens": 256  
  }' | jq .
```

## 📊 Monitoring Your Server

### Real-time Monitoring

```bash
# From your Mac, monitor server in real-time:

# System resources
ssh -p 999 arnav@manthram.tplinkdns.com 'htop'

# Pipeline progress
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/pipeline.log'

# API server logs
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/api.log'

# Training progress
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/training.log | grep -E "(loss|epoch|step)"'
```

### Check Status

```bash
# Quick status check from Mac
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project

echo "=== Service Status ==="
ps aux | grep python | grep -v grep

echo -e "\n=== Disk Usage ==="
df -h . 

echo -e "\n=== Data Files ==="
find data/ -name "*.json*" -exec ls -lh {} \; 2>/dev/null | head -5

echo -e "\n=== Screen Sessions ==="
screen -list 2>/dev/null || echo "No screen sessions"

echo -e "\n=== API Status ==="
curl -s http://127.0.0.1:8000/health 2>/dev/null || echo "API not running"
EOF
```

## 🎯 Production Usage Scenarios

### Scenario 1: Continuous Data Collection

```bash
# Setup continuous data collection on server
ssh -p 999 arnav@manthram.tplinkdns.com

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Start long-running data collection
screen -S data-collector

# Collect large dataset (overnight)
python scripts/pipeline.py --phase data_collection --max-repos 200

# Schedule via cron (optional)
echo "0 2 * * * cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && python scripts/pipeline.py --phase data_collection --max-repos 50" | crontab -
```

### Scenario 2: Training Production Model

```bash
# Run complete training pipeline
ssh -p 999 arnav@manthram.tplinkdns.com

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Production training (several hours)
screen -S production-training

python scripts/pipeline.py --phase full \
  --max-repos 100 \
  --max-iterations 3 \
  --performance-threshold 0.8

# Detach and monitor progress from Mac:
# ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/pipeline.log'
```

### Scenario 3: Production API Service

```bash
# Deploy API as a service
ssh -p 999 arnav@manthram.tplinkdns.com

# Create systemd service
sudo tee /etc/systemd/system/terraform-api.service > /dev/null << 'EOF'
[Unit]
Description=Terraform Prediction API
After=network.target

[Service]
Type=simple
User=arnav
WorkingDirectory=/home/arnav/terraform-prediction-project
Environment=PYTHONPATH=/home/arnav/terraform-prediction-project
EnvironmentFile=/home/arnav/terraform-prediction-project/.env
ExecStart=/home/arnav/terraform-prediction-project/terraform_venv/bin/python server/api.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable terraform-api
sudo systemctl start terraform-api

# Check status
sudo systemctl status terraform-api

# From your Mac, test the API
curl http://manthram.tplinkdns.com:8000/health
```

## 🛠️ Ubuntu-Specific Optimizations

### Memory Optimization

```bash
# Check current memory usage
free -h

# Optimize for your server's RAM
# Edit config/config.py based on available memory:

# For 8GB RAM:
# per_device_train_batch_size = 1
# gradient_accumulation_steps = 16

# For 16GB RAM:
# per_device_train_batch_size = 2  
# gradient_accumulation_steps = 8

# For 32GB+ RAM:
# per_device_train_batch_size = 4
# gradient_accumulation_steps = 4
```

### CPU Optimization

```bash
# Check CPU cores
nproc

# Optimize worker processes
# Edit config/config.py:
# dataloader_num_workers = min(nproc, 8)
```

### Storage Optimization

```bash
# Check disk space
df -h

# Clean up if needed
rm -rf cache/transformers/*  # Clear model cache
rm -f logs/*.log            # Clear old logs
rm -rf data/raw/*           # Clear raw data cache

# Compress old datasets
gzip data/ground_truth/*.json
```

## 🎮 Interactive Server Testing

### Test from Mac Terminal

```bash
# Monitor server interactively
ssh -p 999 arnav@manthram.tplinkdns.com -t 'cd /home/arnav/terraform-prediction-project && bash'

# Run commands on server from Mac
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && python demo_basic.py'

# Check server status
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && screen -list'
```

### Multiple Terminal Workflow

```bash
# Terminal 1: SSH for main commands
ssh -p 999 arnav@manthram.tplinkdns.com
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate && source .env

# Terminal 2: Monitor logs from Mac
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/pipeline.log'

# Terminal 3: Monitor system resources from Mac  
ssh -p 999 arnav@manthram.tplinkdns.com 'watch -n 5 "free -h && df -h /home/arnav"'
```

## ⚡ Ready-to-Run Commands

### Complete Setup Test

Copy and paste this entire block:

```bash
# Deploy and test everything
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
./deploy_to_server.sh

# Test deployment
./server_test_complete.sh
```

### Run Data Collection

```bash
# SSH and run data collection
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate

# Set your GitHub token here
export GITHUB_TOKEN="ghp_your_actual_token"
echo "GITHUB_TOKEN=$GITHUB_TOKEN" >> .env

source .env

# Run data collection
screen -S collection -d -m bash -c '
python scripts/pipeline.py --phase data_collection --max-repos 10 2>&1 | tee collection.log
'

echo "Data collection started in screen session"
echo "Monitor with: screen -r collection"
echo "Or from Mac: ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/collection.log'"
EOF
```

### Start Production API

```bash
# Start API server for production use
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Start API in screen
screen -S terraform-api -d -m python server/api.py --host 0.0.0.0 --port 8000

echo "✅ API server started"
echo "🌐 Access at: http://manthram.tplinkdns.com:8000"
echo "📊 Health check: curl http://manthram.tplinkdns.com:8000/health"
echo "🔍 Attach to screen: screen -r terraform-api"
EOF

# Test from your Mac
sleep 5
curl http://manthram.tplinkdns.com:8000/health | jq .
```

### Full Production Pipeline

```bash
# Run complete production pipeline
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Start full pipeline in screen
screen -S production -d -m bash -c '
python scripts/pipeline.py --phase full \
  --max-repos 50 \
  --max-iterations 3 \
  --performance-threshold 0.8 \
  2>&1 | tee production_pipeline.log
'

echo "🚀 Production pipeline started"
echo "📊 Monitor: screen -r production"
echo "📈 Progress: tail -f production_pipeline.log"
EOF
```

## 🔍 Debugging on Server

### Common Debug Commands

```bash
# Check if services are running
ssh -p 999 arnav@manthram.tplinkdns.com 'ps aux | grep python'

# Check logs for errors
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -50 /home/arnav/terraform-prediction-project/logs/pipeline.log | grep -E "(ERROR|WARN)"'

# Check disk space
ssh -p 999 arnav@manthram.tplinkdns.com 'df -h'

# Check memory usage
ssh -p 999 arnav@manthram.tplinkdns.com 'free -h'

# Check network connectivity
ssh -p 999 arnav@manthram.tplinkdns.com 'curl -s https://api.github.com/rate_limit | jq .rate'
```

### Fix Common Issues

```bash
# Issue: Out of disk space
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && rm -rf cache/* && du -hs data/ models/'

# Issue: Process stuck
ssh -p 999 arnav@manthram.tplinkdns.com 'pkill -f python && screen -wipe'

# Issue: Git authentication
ssh -p 999 arnav@manthram.tplinkdns.com 'git config --global credential.helper store'

# Issue: Environment not loaded
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && env | grep GITHUB_TOKEN'
```

## 📈 Performance Testing

### Load Testing from Mac

```bash
# Test API performance
echo "Testing API load..."

for i in {1..5}; do
  echo "Request $i:"
  time curl -s -X POST "http://manthram.tplinkdns.com:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"terraform_code\": \"resource \\\"aws_s3_bucket\\\" \\\"test$i\\\" { bucket = \\\"test$i\\\" }\"}" | jq -r '.processing_time'
done
```

### Benchmark Training

```bash
# Test training speed on server
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Quick training benchmark
time python scripts/model_training/trainer.py --max-steps 10 --save-steps 5

echo "Training benchmark completed"
EOF
```

## ✅ Server Validation Checklist

Run through this checklist to ensure everything works:

```bash
# 1. ✅ SSH connection
ssh -p 999 arnav@manthram.tplinkdns.com "echo 'Connection OK'"

# 2. ✅ Project deployed  
ssh -p 999 arnav@manthram.tplinkdns.com "ls -la /home/arnav/terraform-prediction-project/config/"

# 3. ✅ Environment setup
ssh -p 999 arnav@manthram.tplinkdns.com "cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && python --version"

# 4. ✅ Dependencies installed
ssh -p 999 arnav@manthram.tplinkdns.com "cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && python -c 'from config.config import config; print(config.model_name)'"

# 5. ✅ Basic functionality
ssh -p 999 arnav@manthram.tplinkdns.com "cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && python demo_basic.py | tail -5"

# 6. ✅ Data collection (requires GITHUB_TOKEN)
# ssh -p 999 arnav@manthram.tplinkdns.com "cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && python scripts/pipeline.py --phase data_collection --max-repos 1"

# 7. ✅ API server
ssh -p 999 arnav@manthram.tplinkdns.com "cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && timeout 10 python server/api.py --host 127.0.0.1 --port 8001 &" && sleep 2 && curl -s http://manthram.tplinkdns.com:8001/health || echo "API test skipped"
```

## 🎉 Success! What's Next?

After successful setup, you can:

### 1. **Collect Training Data**
```bash
# Large-scale data collection (overnight)
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && screen -S collector -d -m python scripts/pipeline.py --phase data_collection --max-repos 100'
```

### 2. **Train the Model**  
```bash
# Full model training (4-8 hours)
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && screen -S trainer -d -m python scripts/pipeline.py --phase full --max-repos 50 --max-iterations 3'
```

### 3. **Deploy Production API**
```bash
# Production API server
ssh -p 999 arnav@manthram.tplinkdns.com 'sudo systemctl start terraform-api'
curl http://manthram.tplinkdns.com:8000/health
```

### 4. **Monitor Everything**
```bash
# Dashboard command from Mac
watch -n 30 'echo "=== $(date) ===" && curl -s http://manthram.tplinkdns.com:8000/health | jq . && ssh -p 999 arnav@manthram.tplinkdns.com "cd /home/arnav/terraform-prediction-project && find data/ -name \"*.json*\" | wc -l" | sed "s/^/Data files: /"'
```

---

## 🎯 TL;DR - Just Run These 3 Commands

```bash
# 1. Deploy
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project && ./deploy_to_server.sh

# 2. Setup and test  
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && nano .env'
# (Add your GITHUB_TOKEN to .env file)

# 3. Run everything
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && python demo_basic.py && screen -S api -d -m python server/api.py --host 0.0.0.0 --port 8000'

# Test: curl http://manthram.tplinkdns.com:8000/health
```

That's it! 🚀
