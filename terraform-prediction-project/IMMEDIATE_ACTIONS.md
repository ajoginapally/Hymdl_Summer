# 🎯 IMMEDIATE ACTIONS - How to Run Everything

This document tells you exactly what to do right now to get the Terraform prediction model working.

## 🚨 Prerequisites (Do First)

1. **Get GitHub Personal Access Token**
   - Go to: https://github.com/settings/tokens
   - Create new token with `repo` permissions
   - Copy the token (starts with `ghp_`)

2. **Optional: Get Hugging Face Token**
   - Go to: https://huggingface.co/settings/tokens
   - Create new token
   - Copy the token (starts with `hf_`)

## 🚀 Option A: Test Locally on Your Mac (Recommended First)

### Step 1: Basic Local Test (5 minutes)

```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# Run quick setup
./quick_start.sh

# Set your GitHub token
export GITHUB_TOKEN="ghp_your_actual_token_here"

# Test basic functionality
python demo_basic.py
```

### Step 2: Quick Local Pipeline Test (15 minutes)

```bash
# Activate environment
source terraform_venv/bin/activate
source .env

# Run small test
python scripts/pipeline.py --phase data_collection --max-repos 2
python scripts/pipeline.py --phase data_processing

# Check results
ls -la data/ground_truth/
ls -la data/processed/
```

### Step 3: Test API Locally

```bash
# Start API server
python server/api.py --host 127.0.0.1 --port 8000 &

# Test it
curl http://localhost:8000/health
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"terraform_code": "resource \"aws_s3_bucket\" \"test\" { bucket = \"test\" }"}' | jq .

# Stop API
pkill -f "server/api.py"
```

## 🐧 Option B: Deploy to Ubuntu Server

### Step 1: Deploy to Server (5 minutes)

```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# Deploy automatically
./deploy_to_server.sh

# Test deployment
./server_test_complete.sh
```

### Step 2: Complete Setup on Server

```bash
# SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

# Navigate to project
cd /home/arnav/terraform-prediction-project
source terraform_venv/bin/activate

# Add your GitHub token
nano .env
# Add line: GITHUB_TOKEN=ghp_your_actual_token_here

# Load environment and test
source .env
python test_setup.py
python demo_basic.py
```

### Step 3: Run Pipeline on Server

```bash
# Still on server - run data collection
screen -S collection
python scripts/pipeline.py --phase data_collection --max-repos 10
# Ctrl+A, D to detach

# Start API server
screen -S api
python server/api.py --host 0.0.0.0 --port 8000
# Ctrl+A, D to detach

# Test from your Mac
curl http://manthram.tplinkdns.com:8000/health
```

## ⚡ Quick Commands Reference

### Most Important Commands

```bash
# 1. Deploy to server
./deploy_to_server.sh

# 2. SSH to server
ssh -p 999 arnav@manthram.tplinkdns.com

# 3. Activate environment (on server)
cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env

# 4. Test basic functionality (on server)
python demo_basic.py

# 5. Run data collection (on server)
python scripts/pipeline.py --phase data_collection --max-repos 5

# 6. Start API server (on server)
screen -S api -d -m python server/api.py --host 0.0.0.0 --port 8000

# 7. Test API (from Mac)
curl http://manthram.tplinkdns.com:8000/health
```

### Debug Commands

```bash
# Check server status
ssh -p 999 arnav@manthram.tplinkdns.com 'ps aux | grep python'

# View logs
ssh -p 999 arnav@manthram.tplinkdns.com 'tail -f /home/arnav/terraform-prediction-project/logs/pipeline.log'

# Check screen sessions
ssh -p 999 arnav@manthram.tplinkdns.com 'screen -list'

# Restart everything
ssh -p 999 arnav@manthram.tplinkdns.com 'pkill -f python; screen -wipe; cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env'
```

## 🎯 What Each Phase Does

### Data Collection
- **Input**: GitHub repositories
- **Output**: `data/ground_truth/terraform_dataset.json.gz`
- **Time**: 10-60 minutes depending on repo count
- **Command**: `python scripts/pipeline.py --phase data_collection --max-repos 10`

### Data Processing  
- **Input**: Raw collected data
- **Output**: `data/processed/train.jsonl`, `validation.jsonl`, `test.jsonl`
- **Time**: 5-15 minutes
- **Command**: `python scripts/pipeline.py --phase data_processing`

### Model Training
- **Input**: Processed training data
- **Output**: `models/fine_tuned/` directory with trained model
- **Time**: 1-4 hours depending on data size
- **Command**: `python scripts/pipeline.py --phase training`

### API Server
- **Input**: Trained model
- **Output**: REST API at port 8000
- **Time**: Starts immediately
- **Command**: `python server/api.py --host 0.0.0.0 --port 8000`

## 🔥 Ready-to-Copy Commands

### Complete Local Test

```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
export GITHUB_TOKEN="ghp_your_token_here"
./quick_start.sh
source terraform_venv/bin/activate && source .env
python demo_basic.py
python scripts/pipeline.py --phase data_collection --max-repos 2
python scripts/pipeline.py --phase data_processing
ls -la data/processed/
```

### Complete Server Deployment

```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
./deploy_to_server.sh
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && nano .env'
# Add: GITHUB_TOKEN=ghp_your_token_here
ssh -p 999 arnav@manthram.tplinkdns.com 'cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env && python demo_basic.py && screen -S api -d -m python server/api.py --host 0.0.0.0 --port 8000'
curl http://manthram.tplinkdns.com:8000/health
```

### Full Production Pipeline on Server

```bash
ssh -p 999 arnav@manthram.tplinkdns.com
cd /home/arnav/terraform-prediction-project && source terraform_venv/bin/activate && source .env
screen -S production -d -m python scripts/pipeline.py --phase full --max-repos 50 --max-iterations 3 --performance-threshold 0.8
echo "Pipeline started. Monitor with: screen -r production"
```

## ❗ If Something Fails

### Quick Fixes

```bash
# Fix 1: Python path issues
export PYTHONPATH=/Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project  # Local
# OR on server:
export PYTHONPATH=/home/arnav/terraform-prediction-project

# Fix 2: Dependencies missing
pip install -r requirements.txt

# Fix 3: GitHub rate limit
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit

# Fix 4: Terraform not found
brew install terraform  # Mac
sudo apt install terraform  # Ubuntu

# Fix 5: Environment not loaded
source .env  # Make sure .env has GITHUB_TOKEN
```

## 🎉 Success Indicators

You'll know it's working when:

✅ **Local**: `python demo_basic.py` shows all demos passing  
✅ **Server**: `curl http://manthram.tplinkdns.com:8000/health` returns healthy status  
✅ **Data**: `ls -la data/ground_truth/` shows dataset files  
✅ **Training**: `tail logs/training.log` shows training progress  
✅ **API**: Prediction requests return JSON responses  

## 📞 Need Help?

1. **Check the guides**:
   - `HOW_TO_RUN.md` - Detailed local instructions
   - `RUN_ON_UBUNTU.md` - Server-specific guide
   - `SETUP_GUIDE.md` - Comprehensive setup guide

2. **Run diagnostics**:
   ```bash
   python test_setup.py  # Basic test
   python demo_basic.py   # Functionality test
   ```

3. **Check logs**:
   ```bash
   tail -f logs/pipeline.log  # Main pipeline
   tail -f logs/api.log       # API server
   tail -f logs/training.log  # Model training
   ```

---

**Start here:** `./deploy_to_server.sh` 🚀
