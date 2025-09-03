# Makefile for Terraform Prediction Model

.PHONY: help install setup clean test format lint type-check
.PHONY: data-collect data-process train validate serve
.PHONY: docker-build docker-train docker-serve docker-clean
.PHONY: example-pipeline example-api

# Default target
help:
	@echo "Terraform Prediction Model - Available commands:"
	@echo ""
	@echo "Setup and Development:"
	@echo "  install           Install Python dependencies"
	@echo "  setup             Create directories and setup environment"
	@echo "  clean             Clean generated files and caches"
	@echo "  test              Run tests"
	@echo "  format            Format code with black"
	@echo "  lint              Lint code with flake8"
	@echo "  type-check        Run type checking with mypy"
	@echo ""
	@echo "Pipeline Commands:"
	@echo "  data-collect      Run data collection phase"
	@echo "  data-process      Run data processing phase"
	@echo "  train             Run model training phase"
	@echo "  validate          Run model validation phase"
	@echo "  serve             Start API server"
	@echo ""
	@echo "Docker Commands:"
	@echo "  docker-build      Build Docker images"
	@echo "  docker-train      Run training in Docker"
	@echo "  docker-serve      Serve API in Docker"
	@echo "  docker-clean      Clean Docker resources"
	@echo ""
	@echo "Examples:"
	@echo "  example-pipeline  Run small pipeline example"
	@echo "  example-api       Demonstrate API usage"

# Setup and Development
install:
	pip install -r requirements.txt

setup:
	mkdir -p data/{raw,processed,ground_truth}
	mkdir -p models/{checkpoints,fine_tuned}
	mkdir -p logs cache
	@echo "Setup complete. Set GITHUB_TOKEN environment variable for GitHub API access."

clean:
	rm -rf cache/*
	rm -rf logs/*
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete

test:
	pytest tests/ -v --tb=short

format:
	black scripts/ server/ config/ examples/

lint:
	flake8 scripts/ server/ config/ examples/ --max-line-length=100 --ignore=E203,W503

type-check:
	mypy scripts/ server/ config/ --ignore-missing-imports

# Pipeline Commands
data-collect:
	python scripts/pipeline.py --phase data_collection --max-repos 100

data-process:
	python scripts/pipeline.py --phase data_processing

train:
	python scripts/pipeline.py --phase training

validate:
	python scripts/pipeline.py --phase validation

serve:
	python server/api.py --host 0.0.0.0 --port 8000

# Full pipeline with different configurations
pipeline-small:
	python scripts/pipeline.py --phase full --max-repos 25 --max-iterations 2

pipeline-medium:
	python scripts/pipeline.py --phase full --max-repos 100 --max-iterations 3

pipeline-large:
	python scripts/pipeline.py --phase full --max-repos 500 --max-iterations 5

# Docker Commands
docker-build:
	docker-compose build

docker-train:
	docker-compose --profile training up terraform-training

docker-serve:
	docker-compose --profile production up -d terraform-api redis

docker-full:
	docker-compose --profile production --profile monitoring up -d

docker-clean:
	docker-compose down --volumes --remove-orphans
	docker system prune -f

# Examples
example-pipeline:
	python examples/run_example.py --mode pipeline

example-api:
	python examples/run_example.py --mode api

# Monitoring and debugging
logs:
	tail -f logs/pipeline.log

api-logs:
	docker-compose logs -f terraform-api

training-logs:
	docker-compose logs -f terraform-training

# Development utilities
check-gpu:
	python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

check-terraform:
	terraform version

check-env:
	@echo "Environment Check:"
	@echo "GITHUB_TOKEN: $${GITHUB_TOKEN:+set}"
	@echo "HF_TOKEN: $${HF_TOKEN:+set}"
	@echo "WANDB_API_KEY: $${WANDB_API_KEY:+set}"
	@python -c "import torch; print(f'PyTorch: {torch.__version__}')"
	@python -c "import transformers; print(f'Transformers: {transformers.__version__}')"

# Quick start for new users
quickstart: install setup
	@echo ""
	@echo "🚀 Quick Start Complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Set environment variables:"
	@echo "   export GITHUB_TOKEN='your_token'"
	@echo ""
	@echo "2. Run small example:"
	@echo "   make example-pipeline"
	@echo ""
	@echo "3. Or start with data collection:"
	@echo "   make data-collect"

# CI/CD helpers
ci-test: install test lint type-check

ci-build: docker-build

# Model management
save-model:
	@echo "Saving model checkpoint..."
	cp -r models/fine_tuned models/backup_$$(date +%Y%m%d_%H%M%S)

restore-model:
	@echo "Available backups:"
	@ls -la models/backup_* 2>/dev/null || echo "No backups found"

# Performance testing
benchmark:
	python -m pytest tests/test_performance.py -v --benchmark-only

profile:
	python -m cProfile -o profile.stats scripts/pipeline.py --phase data_collection --max-repos 5
	python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumulative'); p.print_stats(20)"
