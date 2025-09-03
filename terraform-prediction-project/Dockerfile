# Multi-stage Dockerfile for Terraform prediction model

# Build stage - includes all dependencies for training/data collection
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Terraform CLI
RUN wget -O terraform.zip https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip \
    && unzip terraform.zip \
    && mv terraform /usr/local/bin/ \
    && chmod +x /usr/local/bin/terraform \
    && rm terraform.zip

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage - minimal image for serving
FROM python:3.11-slim as production

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Terraform from builder
COPY --from=builder /usr/local/bin/terraform /usr/local/bin/terraform

WORKDIR /app

# Copy requirements and install only serving dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    torch \
    transformers \
    peft \
    bitsandbytes \
    accelerate \
    huggingface-hub \
    datasets \
    jsonlines \
    pydantic

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed data/ground_truth \
    models/checkpoints models/fine_tuned \
    logs cache

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_HOME=/app/cache/huggingface

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command
CMD ["python", "server/api.py", "--host", "0.0.0.0", "--port", "8000"]

# Training stage - includes all development dependencies
FROM builder as training

WORKDIR /app

# Copy full application code
COPY . .

# Create directories with proper permissions
RUN mkdir -p data/raw data/processed data/ground_truth \
    models/checkpoints models/fine_tuned \
    logs cache \
    && chmod -R 755 data models logs cache

# Set environment variables
ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/cache/transformers
ENV HF_HOME=/app/cache/huggingface

# Default command for training
CMD ["python", "scripts/pipeline.py", "--phase", "full"]
