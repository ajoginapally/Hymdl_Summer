# 🐧 Ubuntu Server Setup and Usage Guide

This guide shows you how to deploy and run the Terraform prediction model on your Ubuntu server at `manthram.tplinkdns.com`.

## 🚀 Quick Server Deployment

### From Your Mac

```bash
# 1. Navigate to project directory
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# 2. Deploy to server automatically
./deploy_to_server.sh

# 3. SSH to server to complete setup
ssh -p 999 arnav@manthram.tplinkdns.com
```

### On the Ubuntu Server

```bash
# After SSH login
cd /home/arnav/terraform-prediction-project

# Activate environment
source terraform_venv/bin/activate

# Edit environment file with your tokens
nano .env

# Test setup
python test_setup.py

# Run basic demo
python demo_basic.py
```

## 🔧 Manual Ubuntu Server Setup

If the automated deployment doesn't work, follow these manual steps:

### Step 1: Connect and Prepare Server

```bash
# From your Mac, SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

# Update system
sudo apt update && sudo apt upgrade -y

# Install essential packages
sudo apt install -y python3.11 python3.11-venv python3.11-dev \
    git wget unzip curl jq htop build-essential \
    software-properties-common

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
sudo chmod +x /usr/local/bin/terraform
rm terraform_1.6.0_linux_amd64.zip

# Verify installations
python3.11 --version
terraform version
git --version
```

### Step 2: Transfer Project Files

```bash
# Option A: Use the deployment script (recommended)
# From your Mac:
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
./deploy_to_server.sh

# Option B: Manual transfer
# From your Mac:
tar -czf terraform-project.tar.gz \
    --exclude='terraform_venv' \
    --exclude='cache' \
    --exclude='data' \
    --exclude='models' \
    *.py *.md *.txt *.yml Dockerfile Makefile config/ scripts/ server/ examples/ tests/ monitoring/

scp -P 999 terraform-project.tar.gz arnav@manthram.tplinkdns.com:~/

# On server:
cd /home/arnav
mkdir -p terraform-prediction-project
cd terraform-prediction-project
tar -xzf ~/terraform-project.tar.gz
```

### Step 3: Setup Environment on Server

```bash
# On the Ubuntu server
cd /home/arnav/terraform-prediction-project

# Create virtual environment
python3.11 -m venv terraform_venv
source terraform_venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies (may take 15-20 minutes)
pip install -r requirements.txt

# Create directory structure
mkdir -p data/{raw,processed,ground_truth}
mkdir -p models/{checkpoints,fine_tuned}
mkdir -p logs cache tests

# Setup environment file
cp .env.template .env
```

### Step 4: Configure Environment Variables

```bash
# Edit environment file
nano .env

# Add your actual tokens:
# GITHUB_TOKEN=ghp_your_actual_github_token
# HF_TOKEN=hf_your_huggingface_token
# WANDB_API_KEY=your_wandb_key

# Load environment
source .env

# Set PYTHONPATH
export PYTHONPATH=/home/arnav/terraform-prediction-project
echo 'export PYTHONPATH=/home/arnav/terraform-prediction-project' >> ~/.bashrc
```

## 🧪 Testing on Ubuntu Server

### Test 1: Basic Setup Verification

```bash
# SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

# Navigate and activate
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Run comprehensive test
python test_setup.py

# Expected output: Should show passing tests for basic components
```

### Test 2: Component Testing

```bash
# Test individual components
python -c "
from config.config import config
print(f'✅ Config loaded - Model: {config.model_name}')
"

python -c "
from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
analyzer = TerraformAnalyzer()
print('✅ Terraform analyzer initialized')
"

python -c "
from scripts.data_collection.github_collector import GitHubTerraformCollector  
collector = GitHubTerraformCollector()
print('✅ GitHub collector initialized')
"
```

### Test 3: Basic Functionality

```bash
# Run basic demonstrations
python demo_basic.py

# Should show working configuration, utilities, and analysis
```

## 📊 Running Data Collection on Server

### Small Scale Test (10-15 minutes)

```bash
# Activate environment
source terraform_venv/bin/activate
source .env

# Test data collection with minimal repos
python scripts/pipeline.py --phase data_collection --max-repos 3

# Monitor progress
tail -f logs/pipeline.log

# Check results
ls -la data/ground_truth/
```

### Medium Scale Collection (1-2 hours)

```bash
# Run in screen/tmux to prevent disconnection
screen -S terraform-collection

# OR
tmux new-session -d -s terraform-collection

# Inside screen/tmux:
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Run larger collection
python scripts/pipeline.py --phase data_collection --max-repos 25

# Detach from screen: Ctrl+A, D
# Detach from tmux: Ctrl+B, D

# Reattach later:
screen -r terraform-collection
# OR
tmux attach-session -t terraform-collection
```

### Process Collected Data

```bash
# Process the data
python scripts/pipeline.py --phase data_processing

# Check outputs
ls -la data/processed/
cat data/processed/dataset_info.json | jq .
```

## 🤖 Model Training on Ubuntu Server

### GPU Setup (If Available)

```bash
# Check for GPU
nvidia-smi

# Install CUDA toolkit if needed
sudo apt install nvidia-cuda-toolkit

# Verify PyTorch can see GPU
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'GPU name: {torch.cuda.get_device_name(0)}')
"
```

### CPU-Only Training Setup

```bash
# For CPU-only training, modify config
nano config/config.py

# Change these values for CPU training:
# per_device_train_batch_size = 1
# gradient_accumulation_steps = 16  
# fp16 = False
# dataloader_num_workers = 2
```

### Quick Training Test

```bash
# Run in screen session for long operations
screen -S terraform-training

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Test model loading
python -c "
from scripts.model_training.model_setup import TerraformModelSetup
setup = TerraformModelSetup()
model, tokenizer = setup.setup_model_and_tokenizer()
print('✅ Model loaded successfully')
print(f'Model device: {next(model.parameters()).device}')
"

# Run short training test (30 minutes)
python scripts/pipeline.py --phase training --max-iterations 1

# Monitor training
tail -f logs/training.log

# Detach: Ctrl+A, D
```

### Full Training Pipeline

```bash
# Run complete pipeline (several hours)
screen -S terraform-full-pipeline

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Full pipeline with reasonable settings
python scripts/pipeline.py --phase full \
  --max-repos 50 \
  --max-iterations 3 \
  --performance-threshold 0.7

# Monitor in another terminal
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/pipeline.log'
```

## 🌐 API Server on Ubuntu

### Start Production API Server

```bash
# SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Start API server in production mode
screen -S terraform-api

# Inside screen:
python server/api.py --host 0.0.0.0 --port 8000

# Detach: Ctrl+A, D
```

### Test Server from Your Mac

```bash
# From your Mac, test the server
curl http://manthram.tplinkdns.com:8000/health

# Test prediction
curl -X POST "http://manthram.tplinkdns.com:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"aws_s3_bucket\" \"test\" { bucket = \"my-test-bucket\" }",
    "max_tokens": 512
  }' | jq .
```

### Setup API as System Service

```bash
# On Ubuntu server, create systemd service
sudo tee /etc/systemd/system/terraform-api.service > /dev/null << 'EOF'
[Unit]
Description=Terraform Prediction API
After=network.target

[Service]
Type=simple
User=arnav
WorkingDirectory=/home/arnav/terraform-prediction-project
Environment=PYTHONPATH=/home/arnav/terraform-prediction-project
ExecStart=/home/arnav/terraform-prediction-project/terraform_venv/bin/python server/api.py --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Start and enable service
sudo systemctl daemon-reload
sudo systemctl enable terraform-api
sudo systemctl start terraform-api

# Check status
sudo systemctl status terraform-api

# View logs
sudo journalctl -u terraform-api -f
```

## 🐳 Docker Deployment on Ubuntu Server

### Setup Docker

```bash
# Install Docker
sudo apt install -y docker.io docker-compose

# Add user to docker group
sudo usermod -aG docker arnav

# Logout and login again for group changes to take effect
exit
ssh -p 999 arnav@manthram.tplinkdns.com

# Test Docker
docker --version
docker-compose --version
```

### Deploy with Docker

```bash
cd /home/arnav/terraform-prediction-project

# Build images
make docker-build

# Run training (in background)
docker-compose --profile training up -d terraform-training

# Run API server
docker-compose --profile production up -d terraform-api redis

# Check status
docker-compose ps

# View logs
docker-compose logs -f terraform-api
```

## 📊 Monitoring on Ubuntu Server

### System Monitoring

```bash
# Monitor system resources
htop

# Monitor disk usage
df -h

# Monitor memory
free -h

# Monitor GPU (if available)
nvidia-smi -l 5

# Monitor network
sudo netstat -tlnp | grep :8000
```

### Application Monitoring

```bash
# Monitor pipeline logs
tail -f /home/arnav/terraform-prediction-project/logs/pipeline.log

# Monitor API logs
tail -f /home/arnav/terraform-prediction-project/logs/api.log

# Monitor training progress
tail -f /home/arnav/terraform-prediction-project/logs/training.log | grep -E "(epoch|loss|eval)"

# Check data collection progress
watch -n 30 'ls -la /home/arnav/terraform-prediction-project/data/ground_truth/'
```

### Performance Monitoring

```bash
# CPU and memory usage over time
sar -u 1 10  # CPU usage
sar -r 1 10  # Memory usage

# Disk I/O monitoring
iostat -x 1 10

# Process monitoring
watch -n 5 'ps aux | grep python | grep -v grep'
```

## 🎯 Server-Specific Running Commands

### Data Collection Optimized for Server

```bash
# SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Run optimized for server resources
python scripts/pipeline.py --phase data_collection \
  --max-repos 100 \
  | tee data_collection.log

# Run in background with nohup
nohup python scripts/pipeline.py --phase data_collection --max-repos 100 > data_collection.log 2>&1 &

# Monitor progress
tail -f data_collection.log
```

### Training Optimized for Server

```bash
# Check available resources first
free -h
nproc  # Number of CPU cores
nvidia-smi  # GPU info if available

# Run training optimized for your server
screen -S training

# Inside screen:
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Modify config for server if needed
python -c "
import json
config_updates = {
    'per_device_train_batch_size': 4,  # Adjust based on RAM/GPU
    'gradient_accumulation_steps': 4,
    'dataloader_num_workers': 4,  # Based on CPU cores
    'save_steps': 100,
    'logging_steps': 10
}
print('Recommended server config updates:')
for k, v in config_updates.items():
    print(f'  {k} = {v}')
"

# Run training
python scripts/pipeline.py --phase training

# Detach from screen: Ctrl+A, D
```

### API Server for Production

```bash
# Start production API server
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Option 1: Screen session
screen -S terraform-api
python server/api.py --host 0.0.0.0 --port 8000

# Option 2: Background with logs
nohup python server/api.py --host 0.0.0.0 --port 8000 > api_server.log 2>&1 &

# Option 3: Systemd service (see above)
sudo systemctl start terraform-api
```

## 🎮 Server Testing Commands

### Test Server Resources

```bash
# SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

# Check system specs
echo "System Information:"
echo "CPU: $(nproc) cores"
echo "RAM: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Disk: $(df -h / | tail -1 | awk '{print $4}')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'No GPU')"
echo "Python: $(python3.11 --version)"
echo "Terraform: $(terraform version | head -1)"
```

### Test Network Connectivity

```bash
# Test GitHub API access from server
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit

# Test Hugging Face access
python -c "
from huggingface_hub import whoami
try:
    print(f'Hugging Face user: {whoami()}')
except:
    print('Hugging Face access not configured')
"

# Test outbound internet
curl -s https://httpbin.org/ip | jq .
```

### Run Pipeline Tests

```bash
# Quick test (5-10 minutes)
python scripts/pipeline.py --phase data_collection --max-repos 2

# Medium test (30-60 minutes)
python scripts/pipeline.py --phase full --max-repos 10 --max-iterations 1 --performance-threshold 0.5

# Full pipeline test (2-4 hours)
screen -S full-test
python scripts/pipeline.py --phase full --max-repos 25 --max-iterations 2
# Ctrl+A, D to detach
```

## 🔍 Debugging on Ubuntu Server

### Check Logs

```bash
# Pipeline logs
tail -f logs/pipeline.log

# System logs
sudo journalctl -u terraform-api -f  # If using systemd
sudo dmesg | tail  # System messages

# Process logs
ps aux | grep python
```

### Resource Issues

```bash
# If out of memory
free -h
# Reduce batch size in config/config.py

# If out of disk space
df -h
du -hs data/ models/ cache/
# Clean up: rm -rf cache/* logs/*.log

# If network issues
ping 8.8.8.8
curl -I https://github.com
```

### Performance Optimization

```bash
# Check CPU usage
top -p $(pgrep -f python)

# Optimize for your server specs
python -c "
import psutil
print(f'Available RAM: {psutil.virtual_memory().available / 1024**3:.1f} GB')
print(f'CPU count: {psutil.cpu_count()}')
print('Recommended settings:')
print(f'  dataloader_num_workers: {min(psutil.cpu_count(), 8)}')
print(f'  per_device_train_batch_size: {max(1, int(psutil.virtual_memory().available / 1024**3 / 4))}')
"
```

## 🌐 Accessing Server Services

### From Your Mac

```bash
# Test API server
curl http://manthram.tplinkdns.com:8000/health

# Monitor server remotely
ssh -p 999 arnav@manthram.tplinkdns.com 'htop'

# Follow logs remotely
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/pipeline.log'
```

### Port Forwarding (If Needed)

```bash
# Forward server port to your local machine
ssh -p 999 -L 8000:localhost:8000 arnav@manthram.tplinkdns.com

# Then access via: http://localhost:8000
```

## 🚀 Production Server Commands

### Start Full Production Pipeline

```bash
# SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

# Setup for production run
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Run in screen for persistence
screen -S production-pipeline

# Full production pipeline
python scripts/pipeline.py --phase full \
  --max-repos 100 \
  --max-iterations 3 \
  --performance-threshold 0.8 \
  > production_run.log 2>&1

# Detach: Ctrl+A, D
```

### Monitor Production Run

```bash
# From your Mac, monitor remotely
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/production_run.log'

# Check status
ssh -p 999 arnav@manthram.tplinkdns.com 'screen -list'

# Get current stats
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && find data/ -name "*.json*" -exec ls -lh {} \;'
```

### Deploy API for Public Access

```bash
# Setup reverse proxy with nginx (optional)
sudo apt install nginx

# Configure nginx
sudo tee /etc/nginx/sites-available/terraform-api << 'EOF'
server {
    listen 80;
    server_name manthram.tplinkdns.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/terraform-api /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx

# Now API accessible at: http://manthram.tplinkdns.com/
```

## 📋 Server-Specific Optimizations

### For High-RAM Servers (32GB+)

```bash
# Edit config for large datasets
nano config/config.py

# Increase batch sizes:
# per_device_train_batch_size = 8
# gradient_accumulation_steps = 2
# dataloader_num_workers = 8
# max_repos_per_search = 200
```

### For GPU Servers

```bash
# Install CUDA-optimized PyTorch
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Optimize for GPU training
nano config/config.py
# fp16 = True
# per_device_train_batch_size = 8
# gradient_accumulation_steps = 1
```

### For CPU-Only Servers

```bash
# Install CPU-optimized versions
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# CPU optimizations
nano config/config.py
# fp16 = False
# per_device_train_batch_size = 1
# gradient_accumulation_steps = 16
# dataloader_num_workers = 4
```

## ⚡ Quick Server Commands Reference

### Start Everything

```bash
# SSH and start services
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Start API in background
nohup python server/api.py --host 0.0.0.0 --port 8000 > api.log 2>&1 &

echo "API started. Test with: curl http://manthram.tplinkdns.com:8000/health"
EOF
```

### Check Status

```bash
# Quick status check from Mac
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
echo "=== Service Status ==="
ps aux | grep -E "(python.*server|python.*pipeline)" | grep -v grep
echo ""
echo "=== Disk Usage ==="
df -h /home/arnav/terraform-prediction-project
echo ""
echo "=== Recent Logs ==="
tail -5 logs/pipeline.log 2>/dev/null || echo "No pipeline logs"
tail -5 logs/api.log 2>/dev/null || echo "No API logs"
EOF
```

### Stop Everything

```bash
# Stop all services
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
# Stop Python processes
pkill -f "python.*server"
pkill -f "python.*pipeline"

# Stop systemd service if running
sudo systemctl stop terraform-api 2>/dev/null || true

# Stop Docker services
cd /home/arnav/terraform-prediction-project
docker-compose down 2>/dev/null || true

echo "All services stopped"
EOF
```

## 🎪 Server Performance Testing

### Load Testing

```bash
# From your Mac, test server under load
for i in {1..10}; do
  curl -s -X POST "http://manthram.tplinkdns.com:8000/predict" \
    -H "Content-Type: application/json" \
    -d "{\"terraform_code\": \"resource \\\"aws_s3_bucket\\\" \\\"test$i\\\" { bucket = \\\"test$i\\\" }\"}" &
done
wait

echo "Load test completed"
```

### Benchmark Training Speed

```bash
# SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Benchmark training speed
python -c "
import time
from scripts.model_training.model_setup import TerraformModelSetup

start_time = time.time()
setup = TerraformModelSetup()
model, tokenizer = setup.setup_model_and_tokenizer()
load_time = time.time() - start_time

print(f'Model loading time: {load_time:.2f} seconds')

# Test inference speed
test_input = 'resource \"aws_s3_bucket\" \"test\" { bucket = \"test\" }'
inputs = tokenizer(test_input, return_tensors='pt')

start_time = time.time()
outputs = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 50, do_sample=False)
inference_time = time.time() - start_time

print(f'Inference time: {inference_time:.2f} seconds')
"
```

## ✅ Server Success Validation

### Complete Server Test

```bash
# Run this complete test on the server
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

echo "🧪 Running complete server validation..."

# 1. Test setup
python test_setup.py | tail -10

# 2. Test data collection
python scripts/pipeline.py --phase data_collection --max-repos 1

# 3. Check outputs
echo "Files created:"
find data/ -name "*.json*" -exec ls -lh {} \;

# 4. Test API server
python server/api.py --host 127.0.0.1 --port 8001 &
API_PID=$!
sleep 5

# Test API
curl -s http://127.0.0.1:8001/health | jq .

# Cleanup
kill $API_PID

echo "✅ Server validation completed"
EOF
```

## 🎉 Ready-to-Use Server Commands

### Complete Setup and Test

```bash
# 1. Deploy from Mac
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
./deploy_to_server.sh

# 2. Complete setup on server
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate

# Set your GitHub token
export GITHUB_TOKEN="ghp_your_actual_token_here"
echo "GITHUB_TOKEN=ghp_your_actual_token_here" >> .env

source .env

# Test everything
python test_setup.py
python demo_basic.py

# Run small pipeline test
python scripts/pipeline.py --phase data_collection --max-repos 3

echo "🎉 Server setup and test completed!"
EOF
```

### Start Production Services

```bash
# Start API server for production
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Start in screen
screen -S terraform-api -d -m python server/api.py --host 0.0.0.0 --port 8000

echo "API server started in screen session"
echo "Access at: http://manthram.tplinkdns.com:8000"
echo "Attach to screen: screen -r terraform-api"
EOF

# Test from your Mac
curl http://manthram.tplinkdns.com:8000/health
```

## 📞 Getting Help with Server Issues

### Debug Server Problems

```bash
# Check server connectivity
ping manthram.tplinkdns.com
ssh -p 999 -v arnav@manthram.tplinkdns.com  # Verbose SSH

# Check server logs
ssh -p 999 arnav@manthram.tplinkdns.com 'sudo journalctl --since "1 hour ago" | tail -50'

# Check service status
ssh -p 999 arnav@manthram.tplinkdns.com 'sudo systemctl status terraform-api'

# Check open ports
ssh -p 999 arnav@manthram.tplinkdns.com 'sudo netstat -tlnp'
```

### Server Resource Issues

```bash
# Check resource usage
ssh -p 999 arnav@manthram.tplinkdns.com << 'EOF'
echo "=== System Resources ==="
free -h
df -h
ps aux | grep python | grep -v grep
nvidia-smi 2>/dev/null || echo "No GPU available"
EOF
```

---

## 🎯 Complete Server Deployment Checklist

- [ ] SSH connection works: `ssh -p 999 arnav@manthram.tplinkdns.com`
- [ ] Project deployed: `./deploy_to_server.sh`
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Environment configured: `.env` file with tokens
- [ ] Basic tests pass: `python test_setup.py`
- [ ] Data collection works: `--max-repos 1` test
- [ ] API server starts: `python server/api.py`
- [ ] API accessible: `curl http://manthram.tplinkdns.com:8000/health`
- [ ] Full pipeline runs: End-to-end test

**Ready to deploy? Run:**

```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
./deploy_to_server.sh
```

Then follow the on-screen instructions to complete the setup! 🚀
