# 🚀 Local Setup and Testing Guide

This guide will walk you through setting up and testing the Terraform prediction model locally on macOS.

## Prerequisites

### 1. System Requirements
- **macOS** (tested on macOS 12+)
- **Python 3.11+** (recommend using pyenv)
- **Git** (for repository operations)
- **Terraform CLI** (for ground truth generation)
- **8GB+ RAM** (16GB+ recommended for training)
- **GPU** (optional but recommended for training)

### 2. Install System Dependencies

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Terraform
brew install terraform

# Verify Terraform installation
terraform version

# Install Python 3.11 (if not already installed)
brew install python@3.11

# Or use pyenv for Python version management
brew install pyenv
pyenv install 3.11.8
pyenv global 3.11.8
```

## 📦 Project Setup

### 1. Navigate to Project Directory
```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
```

### 2. Create Python Virtual Environment
```bash
# Create virtual environment
python3.11 -m venv terraform_venv

# Activate virtual environment
source terraform_venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### 3. Install Python Dependencies
```bash
# Install all dependencies
pip install -r requirements.txt

# If you encounter issues with torch, install it specifically:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Set Environment Variables
```bash
# Create .env file for secrets
cp .env.template .env

# Edit .env file with your tokens
nano .env
```

Add to `.env` file:
```bash
# Required for GitHub API access
GITHUB_TOKEN=your_github_personal_access_token

# Optional for Azure DevOps
AZURE_DEVOPS_TOKEN=your_azure_devops_token

# Hugging Face token (for model access)
HF_TOKEN=your_huggingface_token

# WandB token (for experiment tracking)
WANDB_API_KEY=your_wandb_token

# Log level
LOG_LEVEL=INFO
```

### 5. Load Environment Variables
```bash
# Export variables for current session
source .env

# Or export manually:
export GITHUB_TOKEN="your_token_here"
export HF_TOKEN="your_hf_token_here"
```

### 6. Verify Setup
```bash
# Check environment setup
make check-env

# Check GPU availability (if you have one)
make check-gpu

# Check Terraform installation
make check-terraform
```

## 🧪 Testing the Pipeline

### Step 1: Test Basic Components

```bash
# Test data collection components
python -c "
from scripts.data_collection.github_collector import GitHubTerraformCollector
from scripts.data_collection.terraform_analyzer import TerraformAnalyzer

print('Testing GitHub collector...')
collector = GitHubTerraformCollector()
print('✅ GitHub collector initialized')

print('Testing Terraform analyzer...')
analyzer = TerraformAnalyzer()
print('✅ Terraform analyzer initialized')
"
```

### Step 2: Test Data Collection (Small Scale)

```bash
# Run data collection with just 5 repositories for testing
python scripts/pipeline.py --phase data_collection --max-repos 5
```

Expected output:
- Creates `data/ground_truth/terraform_dataset.json.gz`
- Shows progress bars for repository discovery and analysis
- Logs repository processing results

### Step 3: Test Data Processing

```bash
# Process the collected data
python scripts/pipeline.py --phase data_processing
```

Expected output:
- Creates `data/processed/train.jsonl`, `validation.jsonl`, `test.jsonl`
- Creates `data/processed/dataset_info.json` with statistics
- Shows tokenization progress

### Step 4: Test Model Setup

```bash
# Test model and tokenizer loading
python -c "
from scripts.model_training.model_setup import TerraformModelSetup

print('Testing model setup...')
setup = TerraformModelSetup()
model, tokenizer = setup.setup_model_and_tokenizer()
print('✅ Model and tokenizer loaded successfully')
print(f'Model type: {type(model)}')
print(f'Tokenizer vocab size: {len(tokenizer)}')
"
```

### Step 5: Test Training (Quick Run)

```bash
# Run a very short training session to test the pipeline
python scripts/model_training/trainer.py --max-steps 10 --save-steps 5
```

### Step 6: Test API Server

```bash
# Start the API server (in a new terminal)
python server/api.py --host 127.0.0.1 --port 8000

# In another terminal, test the API
curl http://localhost:8000/health

# Test a prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-test-bucket\"\n}",
    "max_tokens": 256,
    "temperature": 0.1
  }'
```

## 🏃‍♂️ Running Complete Examples

### Quick Example (Small Scale)
```bash
# Run example with minimal data (5-10 minutes)
python examples/run_example.py --mode pipeline
```

### Medium Scale Test
```bash
# Run with more repositories (30-60 minutes)
make pipeline-small
```

### Full Pipeline Test
```bash
# Run full pipeline (several hours)
make pipeline-medium
```

## 🔍 Debugging and Troubleshooting

### Check Logs
```bash
# View pipeline logs
tail -f logs/pipeline.log

# View API logs  
tail -f logs/api.log

# View training logs
tail -f logs/training.log
```

### Common Issues and Solutions

#### 1. **Import Errors**
```bash
# Make sure PYTHONPATH is set correctly
export PYTHONPATH=/Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# Or add to your shell profile
echo 'export PYTHONPATH=/Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project' >> ~/.zshrc
source ~/.zshrc
```

#### 2. **GitHub Rate Limiting**
```bash
# Check your GitHub API rate limit
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit

# If limited, wait or use a different token
```

#### 3. **Terraform CLI Issues**
```bash
# Make sure Terraform is in PATH
which terraform

# Test Terraform works
terraform version

# If missing, install via Homebrew:
brew install terraform
```

#### 4. **Memory Issues During Training**
```bash
# Reduce batch size in config/config.py
# Edit the file and change:
# per_device_train_batch_size = 2  # Reduce from 4
# gradient_accumulation_steps = 8  # Increase to maintain effective batch size
```

#### 5. **Model Download Issues**
```bash
# Clear transformers cache
rm -rf cache/transformers/*

# Re-download model
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
print('✅ Model downloaded successfully')
"
```

## 📊 Monitoring Progress

### View Dataset Statistics
```bash
# After data processing, view dataset info
cat data/processed/dataset_info.json | jq .

# View dataset summary
cat data/processed/dataset_summary.json | jq .
```

### Monitor Training
```bash
# Watch training logs
tail -f logs/training.log

# Monitor GPU usage (if available)
nvidia-smi -l 1

# Monitor system resources
htop
```

### Check Model Performance
```bash
# After training, run validation
python scripts/pipeline.py --phase validation

# View validation results
cat models/validation_results.json | jq .
```

## 🎯 Production Testing

### 1. Build and Test Docker Images
```bash
# Build images
make docker-build

# Test training in Docker
make docker-train

# Test API in Docker
make docker-serve

# Check API health
curl http://localhost:8000/health
```

### 2. Load Testing the API
```bash
# Install apache bench
brew install httpie

# Test API load
for i in {1..10}; do
  http POST localhost:8000/predict terraform_code="resource \"aws_s3_bucket\" \"test\" { bucket = \"test-$i\" }" &
done
wait
```

## 📈 Performance Expectations

### Local Development (MacBook Pro M1/M2)
- **Data Collection (5 repos)**: 2-5 minutes
- **Data Processing**: 1-2 minutes  
- **Model Training (10 steps)**: 5-10 minutes
- **Validation**: 2-3 minutes
- **API Response Time**: 200-500ms per request

### Training Performance (with GPU)
- **Data Collection (50 repos)**: 30-60 minutes
- **Data Processing**: 5-10 minutes
- **Model Training (1 epoch)**: 1-2 hours
- **Full Pipeline**: 3-4 hours

## 🐛 Debugging Specific Components

### Test Individual Components
```bash
# Test GitHub collector
python -m scripts.data_collection.github_collector

# Test Terraform analyzer
python -c "
from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
analyzer = TerraformAnalyzer()
result = analyzer.analyze_terraform_file('# Simple test\nresource \"aws_s3_bucket\" \"test\" { bucket = \"test\" }')
print(result)
"

# Test tokenizer
python -c "
from scripts.model_training.dataset_processor import TerraformDatasetProcessor
processor = TerraformDatasetProcessor()
print('✅ Dataset processor initialized')
print(f'Tokenizer vocab size: {len(processor.tokenizer)}')
"
```

### Test API Endpoints
```bash
# Test all API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/model-info
curl -X POST http://localhost:8000/reload-model
```

## 🎉 Success Criteria

You'll know the setup is working correctly when:

1. ✅ **Data Collection**: Successfully downloads and analyzes Terraform repositories
2. ✅ **Data Processing**: Creates train/validation/test splits with proper tokenization
3. ✅ **Model Training**: Fine-tunes the model without errors
4. ✅ **API Server**: Responds to health checks and predictions
5. ✅ **End-to-End**: Complete pipeline runs and produces validation metrics

## 📞 Getting Help

If you encounter issues:

1. **Check logs** in the `logs/` directory
2. **Enable debug mode**: `export LOG_LEVEL=DEBUG`
3. **Run smaller examples** first: `make example-pipeline`
4. **Check GitHub issues** for common problems
5. **Verify environment variables** are set correctly

## 🚀 Next Steps

Once basic setup works:

1. **Scale up**: Run with more repositories (`--max-repos 100`)
2. **Customize**: Modify configuration in `config/config.py`
3. **Deploy**: Use Docker for production deployment
4. **Monitor**: Set up WandB for experiment tracking
5. **Extend**: Add support for additional cloud providers

Happy coding! 🎯
