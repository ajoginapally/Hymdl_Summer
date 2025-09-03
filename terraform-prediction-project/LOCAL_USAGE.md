# 🎯 Local Usage Instructions

This guide provides step-by-step instructions for running and testing the Terraform prediction model on your local macOS machine.

## 🚀 Quick Start (5 minutes)

```bash
# 1. Navigate to project directory
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# 2. Run the automated setup script
./quick_start.sh

# 3. Set your GitHub token (required)
export GITHUB_TOKEN="your_github_personal_access_token_here"

# 4. Run the basic test
python test_setup.py

# 5. Run a small example
python examples/run_example.py --mode pipeline
```

## 📋 Detailed Setup Steps

### Step 1: Environment Setup

```bash
# Navigate to project
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# Create and activate virtual environment
python3.11 -m venv terraform_venv
source terraform_venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables

Create a `.env` file with your credentials:

```bash
# Copy template
cp .env.template .env

# Edit with your actual tokens
nano .env
```

Add these variables to `.env`:
```bash
# Required - Get from https://github.com/settings/tokens
GITHUB_TOKEN=ghp_your_actual_github_token_here

# Optional but recommended - Get from https://huggingface.co/settings/tokens  
HF_TOKEN=hf_your_huggingface_token_here

# Optional - For experiment tracking
WANDB_API_KEY=your_wandb_key_here

# Set log level
LOG_LEVEL=INFO
```

Load the environment:
```bash
source .env
```

### Step 3: Verify Setup

```bash
# Run comprehensive setup test
python test_setup.py

# Check individual components
make check-env
make check-terraform
make check-gpu  # If you have GPU
```

## 🧪 Testing Individual Components

### Test 1: Data Collection Components

```bash
# Test GitHub collector
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scripts.data_collection.github_collector import GitHubTerraformCollector
print('Testing GitHub API connection...')
collector = GitHubTerraformCollector()
print('✅ GitHub collector initialized successfully')
"

# Test Terraform analyzer
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
analyzer = TerraformAnalyzer()

test_tf = '''
resource \"aws_s3_bucket\" \"test\" {
  bucket = \"test-bucket\"
  versioning {
    enabled = true
  }
}
'''

result = analyzer.analyze_terraform_content(test_tf)
print('✅ Terraform analyzer working')
print(f'Found {len(result[\"resources\"])} resources')
"
```

### Test 2: Model Components

```bash
# Test tokenizer loading (requires HF_TOKEN)
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

print('Loading Llama tokenizer...')
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-3.2-3B-Instruct',
    trust_remote_code=True
)
print('✅ Tokenizer loaded successfully')
print(f'Vocab size: {len(tokenizer)}')
"

# Test dataset processor
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scripts.model_training.dataset_processor import TerraformDatasetProcessor
processor = TerraformDatasetProcessor()
print('✅ Dataset processor initialized')
"
```

### Test 3: API Server

```bash
# Start server in background
python server/api.py --host 127.0.0.1 --port 8000 &
API_PID=$!

# Wait for server to start
sleep 5

# Test health endpoint
curl -s http://localhost:8000/health | jq .

# Test prediction endpoint (without model)
curl -s -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"aws_s3_bucket\" \"test\" { bucket = \"test\" }"
  }' | jq .

# Stop server
kill $API_PID
```

## 🏃‍♂️ Running the Pipeline

### Quick Test (5-10 minutes)

```bash
# Run with minimal data for testing
python scripts/pipeline.py --phase data_collection --max-repos 3

# Process the collected data  
python scripts/pipeline.py --phase data_processing

# Check what was created
ls -la data/ground_truth/
ls -la data/processed/
```

### Medium Test (30-60 minutes)

```bash
# Run small-scale complete pipeline
python scripts/pipeline.py --phase full --max-repos 10 --max-iterations 1 --performance-threshold 0.5

# Or use the Makefile
make pipeline-small
```

### Full Scale Test (2-4 hours)

```bash
# Run full pipeline with reasonable settings
python scripts/pipeline.py --phase full --max-repos 50 --max-iterations 2 --performance-threshold 0.7

# Or use the Makefile
make pipeline-medium
```

## 📊 Understanding the Output

### Data Collection Output

After data collection, you'll see:
```
data/ground_truth/
├── terraform_dataset.json.gz    # Compressed dataset
└── collection_stats.json        # Statistics about collection

logs/
└── pipeline.log                 # Detailed logs
```

### Data Processing Output

After processing:
```
data/processed/
├── train.jsonl                 # Training samples  
├── validation.jsonl            # Validation samples
├── test.jsonl                  # Test samples
├── dataset_info.json           # Dataset statistics
└── dataset_summary.json        # Human-readable summary
```

### Training Output

After training:
```
models/
├── checkpoints/                # Training checkpoints
├── fine_tuned/                # Final trained model
└── training_results.json      # Training metrics

logs/
└── training.log               # Training logs
```

## 🔍 Monitoring and Debugging

### View Logs in Real-Time

```bash
# Pipeline logs
tail -f logs/pipeline.log

# Training logs
tail -f logs/training.log

# API logs
tail -f logs/api.log
```

### Check Progress

```bash
# View dataset statistics
cat data/processed/dataset_info.json | jq .

# View training results
cat models/training_results.json | jq .

# Check model validation
cat models/validation_results.json | jq .
```

### Debug Common Issues

```bash
# If imports fail, check PYTHONPATH
export PYTHONPATH=/Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# If GitHub rate limited, check limit
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit

# If model download fails, clear cache
rm -rf cache/transformers/*

# If memory issues, check available memory
free -h  # On Linux
vm_stat | head -5  # On macOS
```

## 🎮 Interactive Testing

### Test Individual Functions

```python
# Start Python REPL
python

# Test components interactively
>>> import sys
>>> from pathlib import Path
>>> sys.path.insert(0, str(Path.cwd()))

>>> # Test Terraform analysis
>>> from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
>>> analyzer = TerraformAnalyzer()
>>> 
>>> tf_code = """
... resource "aws_instance" "web" {
...   ami           = "ami-12345"
...   instance_type = "t3.micro"
... }
... """
>>> 
>>> result = analyzer.analyze_terraform_content(tf_code)
>>> print(f"Found {len(result['resources'])} resources")
>>> print(result['resources'][0])

>>> # Test dataset processing
>>> from scripts.model_training.dataset_processor import TerraformDatasetProcessor  
>>> processor = TerraformDatasetProcessor()
>>> print(f"Tokenizer vocabulary size: {len(processor.tokenizer)}")
```

### Test API Interactively

```bash
# Start API server
python server/api.py &

# Test with curl
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n  versioning {\n    enabled = true\n  }\n}",
    "max_tokens": 512
  }' | jq .

# Test with Python requests
python -c "
import requests
import json

response = requests.post(
    'http://localhost:8000/predict',
    json={
        'terraform_code': '''
resource \"azurerm_resource_group\" \"main\" {
  name     = \"example-rg\"
  location = \"East US\"
}
        ''',
        'max_tokens': 256
    }
)

print(json.dumps(response.json(), indent=2))
"
```

## 🎯 Success Validation

### Minimal Success Test

To verify the project works, run this complete test:

```bash
# 1. Activate environment
source terraform_venv/bin/activate
source .env

# 2. Run basic tests
python test_setup.py

# 3. Collect a tiny dataset
python scripts/pipeline.py --phase data_collection --max-repos 2

# 4. Process the data
python scripts/pipeline.py --phase data_processing

# 5. Check outputs exist
ls -la data/processed/

# 6. Start API server and test
python server/api.py --port 8001 &
sleep 3
curl http://localhost:8001/health
pkill -f "server/api.py"
```

### Expected Results

✅ **Data collection**: Downloads 2 repositories and creates `data/ground_truth/terraform_dataset.json.gz`

✅ **Data processing**: Creates `train.jsonl`, `validation.jsonl`, and `test.jsonl` files

✅ **API server**: Responds with healthy status and model loading information

## 🚨 Troubleshooting Common Issues

### Issue 1: Module Not Found
```bash
# Solution
export PYTHONPATH=/Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
```

### Issue 2: GitHub Rate Limiting
```bash
# Check your rate limit
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit

# Solution: Use authenticated token or wait for reset
```

### Issue 3: Terraform Not Found
```bash
# Install Terraform
brew install terraform

# Verify installation
terraform version
```

### Issue 4: Model Download Issues
```bash
# Clear cache and retry
rm -rf cache/transformers/*

# Check Hugging Face access
python -c "from huggingface_hub import whoami; print(whoami())"
```

### Issue 5: Memory Issues
```bash
# Reduce batch size in config/config.py
# Change: per_device_train_batch_size = 1
# Change: gradient_accumulation_steps = 16
```

## 🎨 Customization Options

### Modify Configuration

Edit `config/config.py` to customize:

```python
# Data collection
max_repos_per_search = 50  # Reduce for testing

# Model training  
num_epochs = 1            # Quick training
learning_rate = 5e-4      # Adjust learning rate
per_device_train_batch_size = 2  # Reduce for memory

# Validation
max_validation_samples = 25  # Quick validation
```

### Test Different Providers

```bash
# Focus on AWS repositories
python scripts/data_collection/github_collector.py --provider aws --max-repos 5

# Focus on Azure repositories  
python scripts/data_collection/azure_collector.py --max-repos 5
```

## 📈 Performance Optimization

### For MacBook Pro (M1/M2)

```bash
# Use MPS backend for Apple Silicon
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Optimize memory usage
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

### For Systems with Limited Memory

```bash
# Reduce model precision
export TRANSFORMERS_OFFLINE=1  # Avoid re-downloading

# Use smaller batch sizes
# Edit config/config.py:
# per_device_train_batch_size = 1
# gradient_accumulation_steps = 16
```

## 🎪 Advanced Testing

### Load Testing the API

```bash
# Install testing tools
pip install httpx pytest-asyncio

# Run load test
python -c "
import asyncio
import httpx
import time

async def test_prediction():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            'http://localhost:8000/predict',
            json={'terraform_code': 'resource \"aws_s3_bucket\" \"test\" { bucket = \"test\" }'}
        )
        return response.status_code

async def load_test():
    start = time.time()
    tasks = [test_prediction() for _ in range(10)]
    results = await asyncio.gather(*tasks)
    end = time.time()
    
    print(f'Completed 10 requests in {end-start:.2f}s')
    print(f'Success rate: {results.count(200)}/10')

asyncio.run(load_test())
"
```

### Memory and Performance Monitoring

```bash
# Monitor memory usage during training
python -c "
import psutil
import time

def monitor_resources():
    process = psutil.Process()
    print(f'CPU: {psutil.cpu_percent()}%')
    print(f'Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB')
    print(f'GPU Memory: Check nvidia-smi')

monitor_resources()
"
```

## 🎯 Production-Like Local Testing

### Step 1: Run with Docker

```bash
# Build the images
make docker-build

# Run training in Docker
make docker-train

# Run API in Docker  
make docker-serve

# Test the containerized API
curl http://localhost:8000/health
```

### Step 2: Full Pipeline Simulation

```bash
# Simulate production pipeline with moderate data
python scripts/pipeline.py --phase full \
  --max-repos 25 \
  --max-iterations 2 \
  --performance-threshold 0.6

# Monitor progress
tail -f logs/pipeline.log
```

## 🔄 Iterative Development Workflow

### 1. Code Changes → Test → Deploy

```bash
# After making code changes:

# 1. Run basic tests
python test_setup.py

# 2. Run unit tests  
python -m pytest tests/ -v

# 3. Test specific component
python scripts/data_collection/terraform_analyzer.py

# 4. Test end-to-end with small data
make example-pipeline
```

### 2. Data → Train → Validate Loop

```bash
# Collect fresh data
make data-collect

# Process data
make data-process

# Quick training test
python scripts/model_training/trainer.py --max-steps 50

# Validate results
make validate
```

## 📊 Understanding Results

### Data Collection Results

```bash
# Check what was collected
python -c "
import json, gzip
with gzip.open('data/ground_truth/terraform_dataset.json.gz', 'rt') as f:
    data = json.load(f)
print(f'Total samples: {len(data)}')
print(f'Repositories: {len(set(item[\"repository\"] for item in data))}')
"
```

### Training Results

```bash
# View training metrics
cat models/training_results.json | jq '.train_results'

# View validation metrics
cat models/validation_results.json | jq '.overall_metrics'
```

### API Testing Results

```bash
# Comprehensive API test
python examples/run_example.py --mode api
```

## 🎪 Fun Demonstrations

### Demo 1: Predict Simple Infrastructure

```python
# Create demo_test.py
cat << 'EOF' > demo_test.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Example Terraform configurations
terraform_examples = {
    "simple_s3": """
resource "aws_s3_bucket" "example" {
  bucket = "my-example-bucket"
}
""",
    "vpc_setup": """
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id     = aws_vpc.main.id
  cidr_block = "10.0.1.0/24"
  
  tags = {
    Name = "public-subnet"
  }
}
""",
    "azure_rg": """
resource "azurerm_resource_group" "example" {
  name     = "example-resources"
  location = "East US"
}
"""
}

from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
analyzer = TerraformAnalyzer()

for name, tf_code in terraform_examples.items():
    print(f"\n=== {name.upper()} ===")
    result = analyzer.analyze_terraform_content(tf_code)
    print(f"Resources: {len(result['resources'])}")
    for resource in result['resources']:
        print(f"  - {resource['type']}.{resource['name']}")
EOF

python demo_test.py
rm demo_test.py
```

### Demo 2: End-to-End Mini Pipeline

```bash
# Run complete mini pipeline
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scripts.pipeline import TerraformPredictionPipeline

print('🚀 Running mini pipeline demonstration...')
pipeline = TerraformPredictionPipeline()

# Just test data collection
print('📊 Testing data collection...')
result = pipeline.run_data_collection(max_repos=1)
print(f'Data collection result: {result[\"status\"]}')

if result['status'] == 'success':
    print('📦 Testing data processing...')
    result = pipeline.run_data_processing()
    print(f'Data processing result: {result[\"status\"]}')
"
```

## ✅ Validation Checklist

Before considering the setup complete, verify:

- [ ] Python 3.11+ is installed and working
- [ ] Terraform CLI is available (`terraform version` works)
- [ ] Virtual environment is created and activated
- [ ] All Python dependencies are installed (`pip list` shows transformers, torch, etc.)
- [ ] Environment variables are set (especially `GITHUB_TOKEN`)
- [ ] Directory structure is complete
- [ ] Basic imports work (`python test_setup.py` passes)
- [ ] Data collection can run (`make data-collect` with `--max-repos 1`)
- [ ] API server starts and responds (`curl http://localhost:8000/health`)

## 🚀 Next Steps After Setup

1. **Learn the codebase**: Explore `scripts/` to understand the pipeline
2. **Run examples**: Try different Terraform configurations
3. **Customize**: Modify `config/config.py` for your use case
4. **Scale up**: Increase `max_repos` for larger datasets
5. **Deploy**: Use Docker for production-like testing
6. **Contribute**: Add new cloud providers or improve existing code

## 📞 Getting Help

If something doesn't work:

1. **Check logs**: `tail -f logs/pipeline.log`
2. **Enable debug**: `export LOG_LEVEL=DEBUG`
3. **Run tests**: `python test_setup.py`
4. **Check GitHub issues**: See if others had similar problems
5. **Verify environment**: `make check-env`

Happy testing! 🎉
