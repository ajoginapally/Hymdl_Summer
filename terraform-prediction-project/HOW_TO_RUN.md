# 🚀 How to Run the Terraform Prediction Model

This guide provides the exact commands you need to run to test and use the Terraform prediction model locally.

## ⚡ Super Quick Start (Copy & Paste)

```bash
# 1. Navigate to project
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# 2. Run automated setup
./quick_start.sh

# 3. Set your GitHub token (REQUIRED - get from https://github.com/settings/tokens)
export GITHUB_TOKEN="ghp_your_actual_token_here"

# 4. Activate environment  
source terraform_venv/bin/activate
source .env

# 5. Test basic setup
python test_setup.py

# 6. Run basic demo (works without dependencies)
python demo_basic.py
```

## 🔧 Manual Setup (If Quick Start Fails)

### Step 1: Install Dependencies

```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# Create virtual environment
python3.11 -m venv terraform_venv
source terraform_venv/bin/activate

# Install Python packages
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Install Terraform (if not installed)
brew install terraform
```

### Step 2: Set Environment Variables

```bash
# Copy environment template
cp .env.template .env

# Edit .env file (add your actual tokens)
nano .env

# Load environment variables
source .env

# Or set manually
export GITHUB_TOKEN="your_github_token_here"
export HF_TOKEN="your_huggingface_token_here"  # Optional but recommended
```

### Step 3: Create Directories

```bash
mkdir -p data/{raw,processed,ground_truth}
mkdir -p models/{checkpoints,fine_tuned}
mkdir -p logs cache tests
```

## 🧪 Running Tests and Demos

### Test 1: Basic Setup Verification

```bash
# Test all components
python test_setup.py

# Expected: Some tests may fail if dependencies not installed
# Look for ✅ marks for passing tests
```

### Test 2: Configuration and Utilities Demo

```bash
# Run basic demo (no ML dependencies required)
python demo_basic.py

# Expected: Should show configuration, utilities, and basic analysis working
```

### Test 3: Component Testing

```bash
# Test basic imports (without heavy ML libraries)
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

# Test config
from config.config import config
print('✅ Configuration loaded')

# Test utilities  
from config.utils import setup_logging
print('✅ Utilities loaded')
"
```

## 📊 Running Data Collection

### Minimal Data Collection Test (5 minutes)

```bash
# Make sure you're in the project directory and environment is activated
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
source terraform_venv/bin/activate
source .env

# Test with just 1 repository
python scripts/pipeline.py --phase data_collection --max-repos 1

# Check what was created
ls -la data/ground_truth/
```

### Small Scale Collection (15-30 minutes)

```bash
# Collect from 5 repositories
python scripts/pipeline.py --phase data_collection --max-repos 5

# OR use Makefile
make data-collect
```

### Process Collected Data

```bash
# Process the collected data into training format
python scripts/pipeline.py --phase data_processing

# Check processed outputs
ls -la data/processed/
cat data/processed/dataset_info.json | jq .
```

## 🤖 Model Testing (Optional - Requires GPU/Time)

### Quick Model Test

```bash
# Test model loading (downloads ~3GB model)
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scripts.model_training.model_setup import TerraformModelSetup
setup = TerraformModelSetup()
model, tokenizer = setup.setup_model_and_tokenizer()
print('✅ Model loaded successfully')
"
```

### Quick Training Test (10-30 minutes)

```bash
# Run very short training to test pipeline
python scripts/pipeline.py --phase training --max-iterations 1

# OR test just the trainer
python scripts/model_training/trainer.py --max-steps 10
```

## 🌐 API Server Testing

### Start API Server

```bash
# Terminal 1: Start server
source terraform_venv/bin/activate
source .env
python server/api.py --host 127.0.0.1 --port 8000

# Server should start and show: 
# INFO:     Started server process [PID]
# INFO:     Uvicorn running on http://127.0.0.1:8000
```

### Test API Endpoints

```bash
# Terminal 2: Test endpoints

# Health check
curl http://localhost:8000/health

# Model info (will show degraded until model is trained)
curl http://localhost:8000/model-info

# Try prediction (will fail without trained model)
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"aws_s3_bucket\" \"test\" { bucket = \"test\" }"
  }'
```

## 🎯 End-to-End Examples

### Example 1: Mini Pipeline (30-60 minutes)

```bash
# Complete mini pipeline with minimal data
python scripts/pipeline.py --phase full --max-repos 3 --max-iterations 1 --performance-threshold 0.3

# OR
make pipeline-small
```

### Example 2: API Demo

```bash
# Run API demonstration
python examples/run_example.py --mode api

# Run pipeline demonstration
python examples/run_example.py --mode pipeline
```

## 🔍 Monitoring and Debugging

### View Real-time Logs

```bash
# In separate terminals:

# Pipeline logs
tail -f logs/pipeline.log

# Training logs (when training)
tail -f logs/training.log

# API logs (when API running)
tail -f logs/api.log
```

### Check Progress

```bash
# View what files exist
find data/ -type f -name "*.json*" -o -name "*.jsonl"

# Check dataset statistics
cat data/processed/dataset_info.json | jq .

# Check model directory
ls -la models/
```

## 🚨 Troubleshooting Common Issues

### Issue 1: Dependencies Not Installed

```bash
# Symptoms: ImportError for torch, transformers, etc.
# Solution:
source terraform_venv/bin/activate
pip install -r requirements.txt

# If torch installation fails:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 2: GitHub API Issues

```bash
# Check if token is set
echo $GITHUB_TOKEN

# Test GitHub API access
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/user

# Check rate limits
curl -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit
```

### Issue 3: Import Errors

```bash
# Set Python path
export PYTHONPATH=/Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project

# Add to shell profile for persistence
echo 'export PYTHONPATH=/Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project' >> ~/.zshrc
source ~/.zshrc
```

### Issue 4: Terraform CLI Issues

```bash
# Check Terraform installation
which terraform
terraform version

# Install if missing
brew install terraform

# Verify can run basic commands
terraform -help
```

### Issue 5: Model Download Issues

```bash
# Clear cache
rm -rf cache/transformers/*

# Test Hugging Face access
python -c "
from huggingface_hub import whoami
print(whoami())
"

# Manual model download test
python -c "
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-3B-Instruct')
print('Model downloaded successfully')
"
```

## 📋 Step-by-Step Validation

### Validate Each Component

```bash
# 1. Test Python environment
python --version  # Should be 3.11+

# 2. Test package imports
python -c "import torch, transformers, datasets; print('✅ ML packages OK')"

# 3. Test project imports
python -c "
import sys; sys.path.insert(0, '.')
from config.config import config
from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
print('✅ Project imports OK')
"

# 4. Test Terraform CLI
terraform version

# 5. Test GitHub API
curl -s -H "Authorization: token $GITHUB_TOKEN" https://api.github.com/rate_limit | jq .

# 6. Test data collection
python scripts/pipeline.py --phase data_collection --max-repos 1

# 7. Test data processing
python scripts/pipeline.py --phase data_processing

# 8. Test API server
python server/api.py --port 8001 &
sleep 3
curl http://localhost:8001/health
pkill -f "server/api.py"
```

## 🎮 Interactive Testing

### Python REPL Testing

```python
# Start Python
python

# Test components interactively
>>> import sys
>>> from pathlib import Path  
>>> sys.path.insert(0, str(Path.cwd()))

>>> # Test configuration
>>> from config.config import config
>>> print(f"Model: {config.model_name}")
>>> print(f"Data dir: {config.data_dir}")

>>> # Test Terraform analysis
>>> from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
>>> analyzer = TerraformAnalyzer()
>>> 
>>> result = analyzer.analyze_terraform_content('''
... resource "aws_s3_bucket" "test" {
...   bucket = "my-test-bucket"
... }
... ''')
>>> 
>>> print(f"Resources: {len(result['resources'])}")
>>> print(f"First resource: {result['resources'][0]}")

>>> exit()
```

## 🎪 Working Examples

### Example 1: Analyze Terraform Code

```bash
# Create test Terraform file
cat > test_config.tf << 'EOF'
resource "aws_instance" "web" {
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.micro"
  
  tags = {
    Name = "web-server"
    Environment = "test"
  }
}

resource "aws_security_group" "web" {
  name        = "web-sg"
  description = "Security group for web server"
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
EOF

# Analyze it
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
analyzer = TerraformAnalyzer()

with open('test_config.tf') as f:
    tf_content = f.read()

result = analyzer.analyze_terraform_content(tf_content)
print(f'Found {len(result[\"resources\"])} resources:')
for r in result['resources']:
    print(f'  - {r[\"type\"]}.{r[\"name\"]}')
print(f'Complexity score: {result[\"complexity_score\"]}')
print(f'Provider: {result.get(\"primary_provider\", \"unknown\")}')
"

# Clean up
rm test_config.tf
```

### Example 2: Test Ground Truth Generation

```bash
# Create a simple test repository structure
mkdir -p test_repo
cat > test_repo/main.tf << 'EOF'
resource "aws_s3_bucket" "example" {
  bucket = "test-bucket-12345"
}
EOF

cat > test_repo/variables.tf << 'EOF'
variable "bucket_name" {
  description = "Name of the S3 bucket"
  type        = string
  default     = "test-bucket"
}
EOF

# Test repository analysis
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))

from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
analyzer = TerraformAnalyzer()

result = analyzer.analyze_terraform_directory('test_repo')
print(f'Repository analysis results:')
print(f'  Files: {len(result[\"files\"])}')
print(f'  Resources: {len(result[\"resources\"])}')
print(f'  Variables: {len(result[\"variables\"])}')
print(f'  Complexity: {result[\"complexity_score\"]}')
"

# Clean up
rm -rf test_repo
```

## 🎯 Production Testing Commands

### Full Local Pipeline Test

```bash
# Set up for longer run (1-2 hours)
export GITHUB_TOKEN="your_token_here"
source terraform_venv/bin/activate

# Run medium-scale pipeline
python scripts/pipeline.py --phase full \
  --max-repos 20 \
  --max-iterations 2 \
  --performance-threshold 0.6

# Monitor progress
tail -f logs/pipeline.log
```

### Docker Testing

```bash
# Build Docker images
make docker-build

# Test training container
docker-compose --profile training up terraform-training

# Test API container  
docker-compose --profile production up -d terraform-api

# Test containerized API
curl http://localhost:8000/health

# Clean up
docker-compose down
```

## 📈 Performance Monitoring

### System Resources

```bash
# Monitor during training/collection
# Terminal 1: Run pipeline
python scripts/pipeline.py --phase data_collection --max-repos 10

# Terminal 2: Monitor resources
watch -n 2 'ps aux | grep python'
watch -n 2 'df -h'  # Disk usage
watch -n 2 'free -h'  # Memory usage (Linux)
```

### Application Metrics

```bash
# View logs with timestamps
tail -f logs/pipeline.log | grep -E "(INFO|ERROR|WARNING)"

# Check file sizes
du -hs data/ models/ cache/

# Monitor API performance
curl -w "@-" -s -o /dev/null -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"terraform_code": "resource \"aws_s3_bucket\" \"test\" { bucket = \"test\" }"}' << 'EOF'
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
EOF
```

## 🎪 Fun Interactive Examples

### Example 1: Real-time Terraform Analysis

```bash
# Create interactive analyzer
python -i -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scripts.data_collection.terraform_analyzer import TerraformAnalyzer

analyzer = TerraformAnalyzer()
print('🔍 Interactive Terraform analyzer loaded!')
print('Try: result = analyzer.analyze_terraform_content(your_terraform_code)')
"

# In the Python shell, try:
# result = analyzer.analyze_terraform_content('''
# resource "aws_instance" "test" {
#   ami = "ami-12345"  
#   instance_type = "t3.micro"
# }
# ''')
# print(result)
```

### Example 2: GitHub Repository Explorer

```bash
# Explore GitHub repositories interactively
python -i -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
from scripts.data_collection.github_collector import GitHubTerraformCollector

collector = GitHubTerraformCollector()
print('🌐 GitHub collector loaded!')
print('Try: repos = collector.search_repositories(\"terraform aws\", max_results=5)')
"

# In the Python shell, try:
# repos = collector.search_repositories("terraform aws language:hcl", max_results=3)
# for repo in repos:
#     print(f"{repo['full_name']} - {repo['stargazers_count']} stars")
```

## ✅ Success Criteria

You know everything is working when:

### ✅ Basic Setup Success
```bash
python test_setup.py
# Should show: "🎉 ALL TESTS PASSED!" or mostly passing tests
```

### ✅ Data Collection Success
```bash
python scripts/pipeline.py --phase data_collection --max-repos 1
# Should create: data/ground_truth/terraform_dataset.json.gz
ls -la data/ground_truth/  # File should exist and be > 0 bytes
```

### ✅ Data Processing Success
```bash
python scripts/pipeline.py --phase data_processing
# Should create: data/processed/*.jsonl files
ls -la data/processed/  # Should have train.jsonl, validation.jsonl, test.jsonl
```

### ✅ API Server Success
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy" or "degraded","model_loaded":true/false,...}
```

## 🎮 Next Steps After Basic Testing

### 1. Scale Up Data Collection
```bash
# Collect more repositories for better training
python scripts/pipeline.py --phase data_collection --max-repos 50
```

### 2. Full Training Run
```bash
# Run complete training pipeline (several hours)
python scripts/pipeline.py --phase full --max-repos 25 --max-iterations 3
```

### 3. Production Deployment
```bash
# Deploy with Docker
make docker-serve

# Access at http://localhost:8000
```

### 4. Customize for Your Use Case
```bash
# Edit configuration
nano config/config.py

# Add new providers
# Implement in scripts/data_collection/
```

## 📞 Getting Immediate Help

If you're stuck, try these commands in order:

```bash
# 1. Check environment
echo "GITHUB_TOKEN: ${GITHUB_TOKEN:+SET}"
echo "Python: $(python --version)"
echo "Terraform: $(terraform version | head -1)"

# 2. Test basic imports
python -c "
import sys
print(f'Python path: {sys.path[0]}')
try:
    from config.config import config
    print('✅ Config imports work')
except Exception as e:
    print(f'❌ Config import failed: {e}')
"

# 3. Run minimal test
python demo_basic.py

# 4. Check logs for errors
tail -20 logs/pipeline.log

# 5. Verify file structure
find . -name "*.py" | head -10
```

## 🎉 Celebration Commands

When everything works:

```bash
# Show off your working pipeline!
echo "🎉 Terraform Prediction Model is working!"
echo "📊 Dataset size: $(cat data/processed/dataset_info.json | jq -r '.splits.train.sample_count // 0') training samples"
echo "🤖 Model: $(cat config/config.py | grep 'model_name' | head -1)"
echo "🚀 API: http://localhost:8000"
echo "📖 Logs: tail -f logs/pipeline.log"
```

---

**Ready to start? Just run:**

```bash
cd /Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project
./quick_start.sh
```

Then follow the on-screen instructions! 🚀
