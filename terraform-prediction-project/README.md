# Terraform Prediction Model

A machine learning pipeline for predicting Terraform resource changes using fine-tuned language models.

## Overview

This project implements an end-to-end pipeline that:

1. **Collects** Terraform configurations from GitHub repositories
2. **Generates** ground truth by executing `terraform plan` commands
3. **Processes** datasets with tokenization and stratification
4. **Fine-tunes** Llama-3.2-3B-Instruct model with QLoRA
5. **Validates** model performance against test data
6. **Serves** predictions via REST API

## Project Structure

```
terraform-prediction-project/
├── config/                     # Configuration and utilities
│   ├── config.py              # Main configuration
│   └── utils.py               # Utility functions
├── scripts/                   # Core pipeline components
│   ├── data_collection/       # Data collection scripts
│   │   ├── github_collector.py
│   │   ├── terraform_analyzer.py
│   │   ├── azure_collector.py
│   │   └── ground_truth_generator.py
│   ├── model/                 # Model training components
│   │   ├── dataset_processor.py
│   │   ├── model_setup.py
│   │   └── trainer.py
│   ├── validation/            # Validation and testing
│   │   ├── model_validator.py
│   │   └── error_mitigation.py
│   └── pipeline.py            # Main pipeline orchestrator
├── server/                    # API server
│   └── api.py                # FastAPI application
├── data/                      # Data directories
├── models/                    # Model artifacts
├── logs/                      # Log files
├── monitoring/               # Monitoring configuration
├── Dockerfile               # Container configuration
├── docker-compose.yml       # Service orchestration
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Quick Start

### Local Development

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Set up environment:**
```bash
# Set GitHub token for API access
export GITHUB_TOKEN="your_token_here"

# Create required directories
mkdir -p data/{raw,processed,ground_truth} models/{checkpoints,fine_tuned} logs cache
```

3. **Run the full pipeline:**
```bash
python scripts/pipeline.py --phase full --max-repos 50
```

4. **Start the API server:**
```bash
python server/api.py --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Training pipeline:**
```bash
# Run training in container
docker-compose --profile training up terraform-training
```

2. **Production API:**
```bash
# Start API server
docker-compose --profile production up -d terraform-api redis
```

3. **Full stack with monitoring:**
```bash
# Start all services
docker-compose --profile production --profile monitoring up -d
```

## Pipeline Phases

### 1. Data Collection

Collects Terraform configurations from GitHub and Azure repositories:

```bash
python scripts/pipeline.py --phase data_collection --max-repos 100
```

Features:
- Repository discovery via GitHub/Azure APIs
- Smart filtering based on file patterns and quality
- Repository cloning and caching
- Terraform file analysis and complexity scoring

### 2. Data Processing

Processes raw data into training-ready format:

```bash
python scripts/pipeline.py --phase data_processing
```

Features:
- Tokenization with Llama-3.2-3B-Instruct tokenizer
- Dataset stratification by cloud provider
- Train/validation/test splits with module boundary isolation
- Synthetic sample generation for edge cases

### 3. Model Training

Fine-tunes the base model on Terraform data:

```bash
python scripts/pipeline.py --phase training
```

Features:
- QLoRA parameter-efficient fine-tuning
- 4-bit quantization for memory efficiency
- Gradient checkpointing and accumulation
- WandB integration for experiment tracking
- Early stopping and checkpoint management

### 4. Model Validation

Validates model performance:

```bash
python scripts/pipeline.py --phase validation
```

Features:
- Inference on test dataset
- Ground truth comparison using Terraform CLI
- Comprehensive metrics (precision, recall, F1, accuracy)
- Provider-specific and complexity-based analysis

### 5. Error Mitigation

Improves model performance through data augmentation:

```bash
python scripts/pipeline.py --phase error_mitigation --validation-file results.json
```

Features:
- Error pattern analysis
- Synthetic sample generation
- Iterative improvement cycles
- Performance threshold monitoring

## API Usage

### Start Server

```bash
python server/api.py --host 0.0.0.0 --port 8000
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Predict Resource Changes

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n}",
    "max_tokens": 512,
    "temperature": 0.1
  }'
```

### Validate Prediction

```bash
curl -X POST "http://localhost:8000/validate" \
  -H "Content-Type: application/json" \
  -d '{
    "terraform_code": "resource \"aws_s3_bucket\" \"example\" {\n  bucket = \"my-bucket\"\n}",
    "expected_output": [...]
  }'
```

## Configuration

Key configuration options in `config/config.py`:

```python
# Data collection
max_repos_per_search: 500
min_terraform_files: 3
cache_ttl_hours: 24

# Model training
model_name: "meta-llama/Llama-3.2-3B-Instruct"
max_seq_length: 1024
lora_r: 16
learning_rate: 2e-4
num_epochs: 3

# Validation
performance_threshold: 0.8
max_validation_samples: 100
```

## Requirements

- **Python**: 3.11+
- **GPU**: CUDA-capable GPU with 8GB+ VRAM (for training)
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ for models and data
- **Terraform**: CLI installed and accessible
- **Git**: For repository operations

## Environment Variables

```bash
# Required for GitHub API access
export GITHUB_TOKEN="your_token_here"

# Optional for Azure DevOps
export AZURE_DEVOPS_TOKEN="your_token_here"

# Hugging Face token (for model access)
export HF_TOKEN="your_token_here"

# WandB token (for experiment tracking)
export WANDB_API_KEY="your_key_here"
```

## Development

### Running Tests

```bash
# Install development dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest tests/ -v
```

### Code Quality

```bash
# Format code
black scripts/ server/ config/

# Lint code  
flake8 scripts/ server/ config/

# Type checking
mypy scripts/ server/ config/
```

### Adding New Providers

1. Create collector in `scripts/data_collection/`
2. Update analyzer patterns in `terraform_analyzer.py`
3. Add provider-specific validation in `model_validator.py`
4. Update stratification logic in `dataset_processor.py`

## Monitoring

The project includes monitoring capabilities:

### Metrics

- Model prediction accuracy and latency
- API request/response metrics
- Training progress and loss curves
- Data collection statistics

### Logs

Structured logging to `logs/` directory:
- `pipeline.log`: Pipeline execution logs
- `api.log`: API server logs
- `training.log`: Model training logs

### Prometheus

When running with monitoring profile:
- Prometheus UI: http://localhost:9090
- Metrics endpoint: http://localhost:8000/metrics

## Performance

### Training Performance

On a single GPU (RTX 4090):
- Data collection: ~2 hours for 100 repos
- Dataset processing: ~30 minutes for 10k samples  
- Model training: ~4 hours for 3 epochs
- Validation: ~1 hour for full test set

### Inference Performance

- Single prediction: ~200ms
- Batch predictions (10): ~1.5s
- Memory usage: ~4GB VRAM

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or enable gradient checkpointing
2. **GitHub rate limiting**: Use authenticated requests with higher limits
3. **Terraform CLI errors**: Ensure proper working directory and dependencies
4. **Model loading errors**: Check file paths and available memory

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python scripts/pipeline.py --phase full --max-repos 10
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- Terraform CLI and HCL parser
- GitHub and Azure DevOps APIs
- Meta's Llama model family
