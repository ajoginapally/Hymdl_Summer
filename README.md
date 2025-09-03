# Terraform Infrastructure Prediction Fine-tuning Project

## Project Overview

This project aims to fine-tune a small language model (3B or 7B parameters) to predict Terraform infrastructure outputs, demonstrating that AI agents can replace complex code with intelligent inference. The model will learn to predict what `terraform plan` would output given Terraform configuration files.

## Objective

Prove that we can reduce code complexity by using an AI agent to predict Terraform infrastructure changes without running the full Terraform planning process.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Collection Pipeline](#data-collection-pipeline)
4. [Dataset Preparation](#dataset-preparation)
5. [Model Selection and Setup](#model-selection-and-setup)
6. [Fine-tuning Process](#fine-tuning-process)
7. [Validation and Testing](#validation-and-testing)
8. [Error Mitigation Loop](#error-mitigation-loop)
9. [Project Structure](#project-structure)
10. [Usage Instructions](#usage-instructions)
11. [Remote Server Setup](#remote-server-setup)

## Prerequisites

### Software Requirements
- Python 3.8+
- Git
- Terraform CLI
- CUDA-compatible GPU (for training)
- SSH client

### Python Dependencies
```bash
pip install torch transformers datasets accelerate bitsandbytes peft
pip install requests beautifulsoup4 gitpython
pip install zstandard jsonlines pandas numpy
pip install huggingface-hub wandb
```

### Hardware Requirements
- GPU with at least 16GB VRAM (for 7B model)
- 32GB+ system RAM
- 500GB+ storage for datasets

## Environment Setup

### 1. Clone and Setup Repository Structure
```bash
mkdir terraform-prediction-project
cd terraform-prediction-project
mkdir {data,scripts,models,outputs,logs}
```

### 2. Install Terraform
```bash
# macOS
brew install terraform

# Linux
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
```

## Data Collection Pipeline

### Phase 1: Repository Discovery and Collection

Create `scripts/collect_terraform_repos.py`:

```python
import os
import json
import requests
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict
import time

class TerraformRepoCollector:
    def __init__(self, github_token: str = None):
        self.github_token = github_token
        self.headers = {"Authorization": f"token {github_token}"} if github_token else {}
        self.aws_services = [
            "ec2", "s3", "rds", "lambda", "iam", "vpc", "elb", "autoscaling",
            "cloudwatch", "sns", "sqs", "dynamodb", "cloudfront", "route53",
            "apigateway", "eks", "ecs", "ecr", "elasticache", "redshift",
            "kinesis", "glue", "athena", "emr", "sagemaker", "cognito",
            "secrets-manager", "parameter-store", "cloudformation", "kms",
            "acm", "waf", "shield", "guardduty", "inspector", "config",
            "cloudtrail", "organizations", "backup", "datasync", "transfer",
            "workspaces", "appstream", "workmail", "connect", "pinpoint",
            "ses", "workdocs", "chime", "worklink", "appconfig", "xray"
        ]
    
    def search_repositories(self, query: str, per_page: int = 30, max_pages: int = 10) -> List[Dict]:
        """Search GitHub for Terraform repositories"""
        repos = []
        
        for page in range(1, max_pages + 1):
            url = f"https://api.github.com/search/repositories"
            params = {
                "q": query,
                "sort": "stars",
                "order": "desc",
                "per_page": per_page,
                "page": page
            }
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                break
                
            data = response.json()
            repos.extend(data["items"])
            
            # Rate limiting
            time.sleep(1)
            
        return repos
    
    def clone_repository(self, repo_url: str, target_dir: str) -> bool:
        """Clone a repository to target directory"""
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, target_dir],
                check=True,
                capture_output=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def analyze_terraform_files(self, repo_path: str) -> Dict:
        """Analyze Terraform files in repository"""
        tf_files = list(Path(repo_path).rglob("*.tf"))
        tfvars_files = list(Path(repo_path).rglob("*.tfvars"))
        
        aws_resources = set()
        resource_count = 0
        
        for tf_file in tf_files:
            try:
                content = tf_file.read_text()
                # Simple regex to find AWS resources
                import re
                aws_pattern = r'resource\s+"aws_(\w+)"'
                matches = re.findall(aws_pattern, content)
                aws_resources.update(matches)
                resource_count += len(matches)
            except:
                continue
                
        return {
            "tf_files": len(tf_files),
            "tfvars_files": len(tfvars_files),
            "aws_resources": list(aws_resources),
            "resource_count": resource_count,
            "covers_target_services": len(set(aws_resources) & set(self.aws_services))
        }
    
    def collect_repositories(self, output_file: str = "data/terraform_repos.json"):
        """Main collection method"""
        queries = [
            "terraform aws language:HCL stars:>10",
            "terraform infrastructure language:HCL stars:>5",
            "aws terraform examples language:HCL",
            "terraform modules aws language:HCL"
        ]
        
        all_repos = []
        
        for query in queries:
            print(f"Searching: {query}")
            repos = self.search_repositories(query)
            all_repos.extend(repos)
            time.sleep(2)  # Rate limiting
        
        # Remove duplicates
        unique_repos = {repo["id"]: repo for repo in all_repos}
        repos_list = list(unique_repos.values())
        
        # Filter and analyze repositories
        analyzed_repos = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for i, repo in enumerate(repos_list[:100]):  # Limit to 100 repos
                print(f"Analyzing repo {i+1}/100: {repo['name']}")
                
                repo_dir = os.path.join(temp_dir, f"repo_{i}")
                if self.clone_repository(repo["clone_url"], repo_dir):
                    analysis = self.analyze_terraform_files(repo_dir)
                    
                    if analysis["resource_count"] > 0:  # Only keep repos with resources
                        repo_data = {
                            "name": repo["name"],
                            "full_name": repo["full_name"],
                            "clone_url": repo["clone_url"],
                            "stars": repo["stargazers_count"],
                            "size": repo["size"],
                            "analysis": analysis
                        }
                        analyzed_repos.append(repo_data)
        
        # Sort by complexity and AWS service coverage
        analyzed_repos.sort(
            key=lambda x: (x["analysis"]["covers_target_services"], x["analysis"]["resource_count"]),
            reverse=True
        )
        
        # Save results
        os.makedirs("data", exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(analyzed_repos, f, indent=2)
        
        print(f"Collected {len(analyzed_repos)} repositories")
        return analyzed_repos

if __name__ == "__main__":
    collector = TerraformRepoCollector()
    repos = collector.collect_repositories()
```

### Phase 2: Ground Truth Generation

Create `scripts/generate_ground_truth.py`:

```python
import os
import json
import subprocess
import tempfile
import zstandard as zstd
from pathlib import Path
from typing import Dict, List, Optional
import shutil

class GroundTruthGenerator:
    def __init__(self, repos_file: str = "data/terraform_repos.json"):
        with open(repos_file) as f:
            self.repos = json.load(f)
        self.output_dir = Path("data/ground_truth")
        self.output_dir.mkdir(exist_ok=True)
    
    def clone_repository(self, repo_url: str, target_dir: str) -> bool:
        """Clone repository for processing"""
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, target_dir],
                check=True,
                capture_output=True,
                text=True
            )
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone {repo_url}: {e}")
            return False
    
    def find_terraform_directories(self, repo_path: str) -> List[str]:
        """Find directories containing Terraform files"""
        tf_dirs = []
        for root, dirs, files in os.walk(repo_path):
            if any(f.endswith('.tf') for f in files):
                tf_dirs.append(root)
        return tf_dirs
    
    def run_terraform_commands(self, tf_dir: str) -> Optional[Dict]:
        """Run terraform init, plan, and show commands"""
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tf_dir)
            
            # Initialize Terraform
            init_result = subprocess.run(
                ["terraform", "init", "-backend=false"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if init_result.returncode != 0:
                print(f"Terraform init failed in {tf_dir}")
                return None
            
            # Create plan
            plan_result = subprocess.run(
                ["terraform", "plan", "-out=tfplan"],
                capture_output=True,
                text=True,
                timeout=600
            )
            
            if plan_result.returncode != 0:
                print(f"Terraform plan failed in {tf_dir}")
                return None
            
            # Generate JSON output
            show_result = subprocess.run(
                ["terraform", "show", "-json", "tfplan"],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if show_result.returncode != 0:
                print(f"Terraform show failed in {tf_dir}")
                return None
            
            return json.loads(show_result.stdout)
            
        except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
            print(f"Error processing {tf_dir}: {e}")
            return None
        finally:
            os.chdir(original_cwd)
    
    def extract_resource_changes(self, plan_data: Dict) -> List[Dict]:
        """Extract and clean resource changes from plan"""
        if "resource_changes" not in plan_data:
            return []
        
        cleaned_changes = []
        
        for change in plan_data["resource_changes"]:
            if change.get("change", {}).get("after") is not None:
                after_state = change["change"]["after"]
                
                # Skip resources with unknown values
                if self.has_unknown_values(after_state):
                    continue
                
                cleaned_change = {
                    "address": change["address"],
                    "type": change["type"],
                    "provider_name": change["provider_name"],
                    "change": {
                        "actions": change["change"]["actions"],
                        "after": after_state
                    }
                }
                cleaned_changes.append(cleaned_change)
        
        return cleaned_changes
    
    def has_unknown_values(self, obj, depth: int = 0) -> bool:
        """Check if object contains unknown values (recursive)"""
        if depth > 10:  # Prevent infinite recursion
            return False
            
        if isinstance(obj, dict):
            for value in obj.values():
                if value == "(known after apply)" or self.has_unknown_values(value, depth + 1):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if self.has_unknown_values(item, depth + 1):
                    return True
        elif isinstance(obj, str) and "(known after apply)" in obj:
            return True
            
        return False
    
    def collect_terraform_files(self, tf_dir: str) -> str:
        """Concatenate all Terraform files into single text block"""
        content_parts = []
        
        # Collect .tf files
        for tf_file in Path(tf_dir).glob("*.tf"):
            try:
                file_content = tf_file.read_text()
                # Strip comments and normalize whitespace
                cleaned_content = self.clean_terraform_content(file_content)
                content_parts.append(f"# File: {tf_file.name}\n{cleaned_content}")
            except Exception as e:
                print(f"Error reading {tf_file}: {e}")
        
        # Collect .tfvars files
        for tfvars_file in Path(tf_dir).glob("*.tfvars"):
            try:
                file_content = tfvars_file.read_text()
                cleaned_content = self.clean_terraform_content(file_content)
                content_parts.append(f"# Variables: {tfvars_file.name}\n{cleaned_content}")
            except Exception as e:
                print(f"Error reading {tfvars_file}: {e}")
        
        return "\n\n".join(content_parts)
    
    def clean_terraform_content(self, content: str) -> str:
        """Clean Terraform content by removing comments and normalizing whitespace"""
        lines = []
        for line in content.split('\n'):
            # Remove comments but preserve structure
            if line.strip().startswith('#'):
                continue
            if '//' in line:
                line = line.split('//')[0].rstrip()
            
            # Preserve non-empty lines and normalize whitespace
            if line.strip():
                lines.append(line.rstrip())
        
        return '\n'.join(lines)
    
    def compress_data(self, data: str) -> bytes:
        """Compress data using Zstandard"""
        cctx = zstd.ZstdCompressor()
        return cctx.compress(data.encode('utf-8'))
    
    def decompress_data(self, compressed_data: bytes) -> str:
        """Decompress Zstandard data"""
        dctx = zstd.ZstdDecompressor()
        return dctx.decompress(compressed_data).decode('utf-8')
    
    def process_repository(self, repo_data: Dict) -> List[Dict]:
        """Process a single repository and generate training samples"""
        repo_name = repo_data["full_name"].replace("/", "_")
        print(f"Processing repository: {repo_data['full_name']}")
        
        samples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = os.path.join(temp_dir, "repo")
            
            if not self.clone_repository(repo_data["clone_url"], repo_dir):
                return samples
            
            # Find all Terraform directories
            tf_dirs = self.find_terraform_directories(repo_dir)
            
            for tf_dir in tf_dirs:
                print(f"  Processing directory: {tf_dir}")
                
                # Get Terraform plan output
                plan_data = self.run_terraform_commands(tf_dir)
                if not plan_data:
                    continue
                
                # Extract resource changes
                resource_changes = self.extract_resource_changes(plan_data)
                if not resource_changes:
                    continue
                
                # Collect input files
                input_content = self.collect_terraform_files(tf_dir)
                if not input_content.strip():
                    continue
                
                # Create training sample
                sample = {
                    "repository": repo_data["full_name"],
                    "directory": os.path.relpath(tf_dir, repo_dir),
                    "input": input_content,
                    "output": json.dumps(resource_changes, sort_keys=True),
                    "metadata": {
                        "resource_count": len(resource_changes),
                        "aws_services": list(set(
                            change["type"].replace("aws_", "") 
                            for change in resource_changes 
                            if change["type"].startswith("aws_")
                        ))
                    }
                }
                
                samples.append(sample)
        
        return samples
    
    def generate_ground_truth_dataset(self, max_repos: int = 50):
        """Generate complete ground truth dataset"""
        all_samples = []
        
        # Select diverse repositories
        selected_repos = self.select_diverse_repositories(max_repos)
        
        for i, repo in enumerate(selected_repos):
            print(f"Processing repo {i+1}/{len(selected_repos)}")
            samples = self.process_repository(repo)
            all_samples.extend(samples)
            
            # Save intermediate results
            if (i + 1) % 5 == 0:
                self.save_samples(all_samples, f"data/ground_truth/samples_checkpoint_{i+1}.jsonl")
        
        # Save final dataset
        self.save_samples(all_samples, "data/ground_truth/complete_dataset.jsonl")
        
        # Generate statistics
        self.generate_dataset_statistics(all_samples)
        
        return all_samples
    
    def select_diverse_repositories(self, max_repos: int) -> List[Dict]:
        """Select repositories with diverse AWS service coverage"""
        # Sort by AWS service coverage and resource count
        sorted_repos = sorted(
            self.repos,
            key=lambda x: (
                x["analysis"]["covers_target_services"],
                x["analysis"]["resource_count"],
                x["stars"]
            ),
            reverse=True
        )
        
        return sorted_repos[:max_repos]
    
    def save_samples(self, samples: List[Dict], filename: str):
        """Save samples to JSONL format"""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
    
    def generate_dataset_statistics(self, samples: List[Dict]):
        """Generate and save dataset statistics"""
        stats = {
            "total_samples": len(samples),
            "total_repositories": len(set(s["repository"] for s in samples)),
            "aws_services_covered": len(set(
                service 
                for s in samples 
                for service in s["metadata"]["aws_services"]
            )),
            "resource_distribution": {},
            "service_distribution": {}
        }
        
        # Calculate distributions
        for sample in samples:
            count = sample["metadata"]["resource_count"]
            bucket = f"{count//10*10}-{count//10*10+9}"
            stats["resource_distribution"][bucket] = stats["resource_distribution"].get(bucket, 0) + 1
            
            for service in sample["metadata"]["aws_services"]:
                stats["service_distribution"][service] = stats["service_distribution"].get(service, 0) + 1
        
        with open("data/ground_truth/dataset_statistics.json", "w") as f:
            json.dump(stats, f, indent=2)
        
        print(f"Dataset Statistics:")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Repositories: {stats['total_repositories']}")
        print(f"  AWS services covered: {stats['aws_services_covered']}")

if __name__ == "__main__":
    # Set GitHub token as environment variable: export GITHUB_TOKEN=your_token
    github_token = os.getenv("GITHUB_TOKEN")
    
    collector = TerraformRepoCollector(github_token)
    
    # First collect repository metadata
    repos = collector.collect_repositories()
    
    # Then generate ground truth dataset
    generator = GroundTruthGenerator()
    samples = generator.generate_ground_truth_dataset(max_repos=50)
```

## Dataset Preparation

Create `scripts/prepare_dataset.py`:

```python
import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Tuple
from transformers import AutoTokenizer
import random
import zstandard as zstd

class DatasetPreparator:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def load_samples(self, filepath: str) -> List[Dict]:
        """Load samples from JSONL file"""
        samples = []
        with jsonlines.open(filepath) as reader:
            for sample in reader:
                samples.append(sample)
        return samples
    
    def create_training_format(self, sample: Dict) -> Dict:
        """Convert sample to training format"""
        # Create instruction-following format
        instruction = """Given the following Terraform configuration, predict the resource changes that would result from running 'terraform plan'. Output the result as JSON containing the resource_changes array with each resource's final state after applying the plan.

Terraform Configuration:
"""
        
        input_text = instruction + sample["input"]
        output_text = sample["output"]
        
        # Tokenize to check length
        input_tokens = self.tokenizer.encode(input_text)
        output_tokens = self.tokenizer.encode(output_text)
        
        if len(input_tokens) + len(output_tokens) > 30000:  # Leave room for context
            return None
        
        return {
            "input": input_text,
            "output": output_text,
            "metadata": sample["metadata"],
            "repository": sample["repository"]
        }
    
    def stratify_by_provider(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Stratify samples by cloud provider"""
        provider_samples = {"aws": [], "mixed": [], "other": []}
        
        for sample in samples:
            aws_services = sample["metadata"].get("aws_services", [])
            if len(aws_services) > 0:
                if len(aws_services) >= 3:
                    provider_samples["aws"].append(sample)
                else:
                    provider_samples["mixed"].append(sample)
            else:
                provider_samples["other"].append(sample)
        
        return provider_samples
    
    def create_train_val_test_split(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create 80/10/10 train/validation/test split by repository boundary"""
        # Group by repository
        repo_samples = {}
        for sample in samples:
            repo = sample["repository"]
            if repo not in repo_samples:
                repo_samples[repo] = []
            repo_samples[repo].append(sample)
        
        # Split repositories
        repos = list(repo_samples.keys())
        random.shuffle(repos)
        
        total_repos = len(repos)
        train_split = int(0.8 * total_repos)
        val_split = int(0.9 * total_repos)
        
        train_repos = repos[:train_split]
        val_repos = repos[train_split:val_split]
        test_repos = repos[val_split:]
        
        # Collect samples
        train_samples = []
        val_samples = []
        test_samples = []
        
        for repo in train_repos:
            train_samples.extend(repo_samples[repo])
        for repo in val_repos:
            val_samples.extend(repo_samples[repo])
        for repo in test_repos:
            test_samples.extend(repo_samples[repo])
        
        return train_samples, val_samples, test_samples
    
    def save_dataset_splits(self, train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
        """Save dataset splits to files"""
        output_dir = Path("data/processed")
        output_dir.mkdir(exist_ok=True)
        
        datasets = {
            "train": train_samples,
            "validation": val_samples,
            "test": test_samples
        }
        
        for split_name, samples in datasets.items():
            filepath = output_dir / f"{split_name}.jsonl"
            with jsonlines.open(filepath, 'w') as writer:
                for sample in samples:
                    formatted_sample = self.create_training_format(sample)
                    if formatted_sample:
                        writer.write(formatted_sample)
            
            print(f"Saved {len(samples)} samples to {filepath}")
    
    def generate_synthetic_examples(self) -> List[Dict]:
        """Generate synthetic examples for under-represented constructs"""
        synthetic_examples = []
        
        # Dynamic block examples
        dynamic_example = {
            "input": """
# File: main.tf
resource "aws_security_group" "example" {
  name_prefix = "example-"
  
  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }
}

# Variables: terraform.tfvars
ingress_rules = [
  {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  },
  {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
]
""",
            "output": json.dumps([{
                "address": "aws_security_group.example",
                "type": "aws_security_group",
                "provider_name": "registry.terraform.io/hashicorp/aws",
                "change": {
                    "actions": ["create"],
                    "after": {
                        "name_prefix": "example-",
                        "ingress": [
                            {
                                "from_port": 80,
                                "to_port": 80,
                                "protocol": "tcp",
                                "cidr_blocks": ["0.0.0.0/0"]
                            },
                            {
                                "from_port": 443,
                                "to_port": 443,
                                "protocol": "tcp",
                                "cidr_blocks": ["0.0.0.0/0"]
                            }
                        ]
                    }
                }
            }]),
            "metadata": {
                "resource_count": 1,
                "aws_services": ["security_group"],
                "has_dynamic_blocks": True
            },
            "repository": "synthetic/dynamic_example"
        }
        
        # Count example
        count_example = {
            "input": """
# File: main.tf
resource "aws_instance" "web" {
  count         = var.instance_count
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.micro"
  
  tags = {
    Name = "web-server-${count.index + 1}"
  }
}

# Variables: terraform.tfvars
instance_count = 3
""",
            "output": json.dumps([
                {
                    "address": f"aws_instance.web[{i}]",
                    "type": "aws_instance",
                    "provider_name": "registry.terraform.io/hashicorp/aws",
                    "change": {
                        "actions": ["create"],
                        "after": {
                            "ami": "ami-0c02fb55956c7d316",
                            "instance_type": "t3.micro",
                            "tags": {"Name": f"web-server-{i+1}"}
                        }
                    }
                } for i in range(3)
            ]),
            "metadata": {
                "resource_count": 3,
                "aws_services": ["instance"],
                "has_count": True
            },
            "repository": "synthetic/count_example"
        }
        
        synthetic_examples.extend([dynamic_example, count_example])
        return synthetic_examples
    
    def prepare_complete_dataset(self):
        """Main method to prepare the complete dataset"""
        print("Loading ground truth samples...")
        samples = self.load_samples("data/ground_truth/complete_dataset.jsonl")
        
        print("Adding synthetic examples...")
        synthetic_samples = self.generate_synthetic_examples()
        samples.extend(synthetic_samples)
        
        print("Stratifying by provider...")
        stratified = self.stratify_by_provider(samples)
        
        # Focus on AWS samples for initial training
        aws_samples = stratified["aws"]
        mixed_samples = stratified["mixed"][:len(aws_samples)//4]  # Add some mixed samples
        
        final_samples = aws_samples + mixed_samples
        random.shuffle(final_samples)
        
        print("Creating train/validation/test splits...")
        train_samples, val_samples, test_samples = self.create_train_val_test_split(final_samples)
        
        print("Saving dataset splits...")
        self.save_dataset_splits(train_samples, val_samples, test_samples)
        
        print(f"Dataset preparation complete!")
        print(f"  Training samples: {len(train_samples)}")
        print(f"  Validation samples: {len(val_samples)}")
        print(f"  Test samples: {len(test_samples)}")

if __name__ == "__main__":
    preparator = DatasetPreparator()
    preparator.prepare_complete_dataset()
```

## Model Selection and Setup

Create `scripts/setup_model.py`:

```python
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training
)

class ModelSetup:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def setup_quantization_config(self):
        """Setup 4-bit quantization configuration"""
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    def setup_lora_config(self):
        """Setup LoRA configuration"""
        return LoraConfig(
            r=16,  # Low-rank adaptation rank
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with quantization"""
        print(f"Loading model: {self.model_name}")
        
        # Setup quantization
        bnb_config = self.setup_quantization_config()
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Prepare for training
        model = prepare_model_for_kbit_training(model)
        
        # Setup LoRA
        lora_config = self.setup_lora_config()
        model = get_peft_model(model, lora_config)
        
        print(f"Model loaded successfully on {self.device}")
        print(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")
        
        return model, tokenizer
    
    def setup_training_arguments(self, output_dir: str = "models/terraform-predictor"):
        """Setup training arguments"""
        return TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,  # Effective batch size: 8
            num_train_epochs=1,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            weight_decay=0.0,
            optim="adamw_torch",
            adam_beta1=0.9,
            adam_beta2=0.95,
            adam_epsilon=1e-8,
            bf16=True,
            dataloader_drop_last=True,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            logging_strategy="steps",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            early_stopping_patience=2,
            report_to="wandb",
            run_name="terraform-predictor-finetuning"
        )

if __name__ == "__main__":
    setup = ModelSetup()
    model, tokenizer = setup.load_model_and_tokenizer()
    training_args = setup.setup_training_arguments()
    
    print("Model setup complete!")
```

## Fine-tuning Process

Create `scripts/fine_tune_model.py`:

```python
import json
import jsonlines
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments
from scripts.setup_model import ModelSetup
import wandb

class TerraformTrainer:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-hf"):
        self.setup = ModelSetup(model_name)
        self.model, self.tokenizer = self.setup.load_model_and_tokenizer()
        
    def load_dataset(self, filepath: str) -> Dataset:
        """Load and tokenize dataset"""
        samples = []
        with jsonlines.open(filepath) as reader:
            for sample in reader:
                samples.append(sample)
        
        def tokenize_function(examples):
            # Combine input and output for causal LM training
            full_texts = []
            for inp, out in zip(examples["input"], examples["output"]):
                full_text = f"{inp}\n\nOUTPUT:\n{out}{self.tokenizer.eos_token}"
                full_texts.append(full_text)
            
            # Tokenize
            tokenized = self.tokenizer(
                full_texts,
                truncation=True,
                padding=True,
                max_length=32768,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            
            return tokenized
        
        # Convert to HuggingFace dataset
        dataset_dict = {
            "input": [s["input"] for s in samples],
            "output": [s["output"] for s in samples]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self):
        """Main training loop"""
        print("Loading datasets...")
        train_dataset = self.load_dataset("data/processed/train.jsonl")
        val_dataset = self.load_dataset("data/processed/validation.jsonl")
        
        print("Setting up training arguments...")
        training_args = self.setup.setup_training_arguments()
        
        print("Initializing trainer...")
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
        )
        
        print("Starting training...")
        trainer.train()
        
        print("Saving final model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(training_args.output_dir)
        
        print("Training complete!")

if __name__ == "__main__":
    # Initialize wandb for experiment tracking
    wandb.init(project="terraform-prediction", name="llama-7b-finetuning")
    
    trainer = TerraformTrainer()
    trainer.train()
```

## Validation and Testing

Create `scripts/validate_model.py`:

```python
import json
import jsonlines
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import subprocess
import tempfile
import os
from pathlib import Path
from typing import Dict, List
import difflib

class ModelValidator:
    def __init__(self, model_path: str = "models/terraform-predictor"):
        self.model_path = model_path
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model"""
        print("Loading fine-tuned model...")
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        self.model.eval()
        
    def predict_terraform_output(self, terraform_config: str) -> str:
        """Generate prediction for Terraform configuration"""
        instruction = """Given the following Terraform configuration, predict the resource changes that would result from running 'terraform plan'. Output the result as JSON containing the resource_changes array with each resource's final state after applying the plan.

Terraform Configuration:
"""
        
        prompt = instruction + terraform_config + "\n\nOUTPUT:\n"
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the generated part
        generated_text = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
        
        # Extract JSON from output
        try:
            # Find JSON content
            if "OUTPUT:" in generated_text:
                json_part = generated_text.split("OUTPUT:")[-1].strip()
            else:
                json_part = generated_text.strip()
            
            # Try to parse as JSON
            parsed = json.loads(json_part)
            return json.dumps(parsed, indent=2)
        except json.JSONDecodeError:
            return generated_text.strip()
    
    def run_terraform_ground_truth(self, terraform_config: str) -> str:
        """Generate ground truth using actual Terraform commands"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Write Terraform files
            config_lines = terraform_config.split('\n')
            current_file = None
            current_content = []
            
            for line in config_lines:
                if line.startswith("# File:"):
                    if current_file and current_content:
                        file_path = os.path.join(temp_dir, current_file)
                        with open(file_path, 'w') as f:
                            f.write('\n'.join(current_content))
                    
                    current_file = line.replace("# File:", "").strip()
                    current_content = []
                elif line.startswith("# Variables:"):
                    if current_file and current_content:
                        file_path = os.path.join(temp_dir, current_file)
                        with open(file_path, 'w') as f:
                            f.write('\n'.join(current_content))
                    
                    current_file = line.replace("# Variables:", "").strip()
                    current_content = []
                else:
                    current_content.append(line)
            
            # Write last file
            if current_file and current_content:
                file_path = os.path.join(temp_dir, current_file)
                with open(file_path, 'w') as f:
                    f.write('\n'.join(current_content))
            
            # Run Terraform commands
            original_cwd = os.getcwd()
            try:
                os.chdir(temp_dir)
                
                # Initialize
                subprocess.run(["terraform", "init", "-backend=false"], 
                             check=True, capture_output=True)
                
                # Plan
                subprocess.run(["terraform", "plan", "-out=tfplan"], 
                             check=True, capture_output=True)
                
                # Show JSON
                result = subprocess.run(["terraform", "show", "-json", "tfplan"], 
                                      check=True, capture_output=True, text=True)
                
                plan_data = json.loads(result.stdout)
                resource_changes = []
                
                for change in plan_data.get("resource_changes", []):
                    if change.get("change", {}).get("after") is not None:
                        resource_changes.append({
                            "address": change["address"],
                            "type": change["type"],
                            "provider_name": change["provider_name"],
                            "change": {
                                "actions": change["change"]["actions"],
                                "after": change["change"]["after"]
                            }
                        })
                
                return json.dumps(resource_changes, indent=2)
                
            except subprocess.CalledProcessError as e:
                return f"Terraform error: {e}"
            finally:
                os.chdir(original_cwd)
    
    def compare_outputs(self, predicted: str, ground_truth: str) -> Dict:
        """Compare predicted output with ground truth"""
        try:
            pred_json = json.loads(predicted)
            truth_json = json.loads(ground_truth)
            
            # Compare as sets of resource addresses
            pred_addresses = set(res["address"] for res in pred_json if isinstance(pred_json, list))
            truth_addresses = set(res["address"] for res in truth_json if isinstance(truth_json, list))
            
            correct = len(pred_addresses & truth_addresses)
            total_truth = len(truth_addresses)
            total_pred = len(pred_addresses)
            
            precision = correct / total_pred if total_pred > 0 else 0
            recall = correct / total_truth if total_truth > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "correct_predictions": correct,
                "total_ground_truth": total_truth,
                "total_predictions": total_pred,
                "missing_resources": list(truth_addresses - pred_addresses),
                "extra_resources": list(pred_addresses - truth_addresses)
            }
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            return {
                "error": str(e),
                "precision": 0,
                "recall": 0,
                "f1_score": 0
            }
    
    def validate_on_test_set(self):
        """Validate model on test dataset"""
        print("Loading test dataset...")
        test_samples = []
        with jsonlines.open("data/processed/test.jsonl") as reader:
            for sample in reader:
                test_samples.append(sample)
        
        results = []
        total_f1 = 0
        
        for i, sample in enumerate(test_samples[:20]):  # Test on subset first
            print(f"Testing sample {i+1}/20...")
            
            # Get model prediction
            predicted = self.predict_terraform_output(sample["input"])
            
            # Compare with ground truth
            comparison = self.compare_outputs(predicted, sample["output"])
            
            result = {
                "sample_id": i,
                "repository": sample.get("repository", "unknown"),
                "prediction": predicted,
                "ground_truth": sample["output"],
                "metrics": comparison
            }
            
            results.append(result)
            total_f1 += comparison.get("f1_score", 0)
            
            print(f"  F1 Score: {comparison.get('f1_score', 0):.3f}")
        
        # Save results
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/validation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        avg_f1 = total_f1 / len(results)
        print(f"\nValidation Complete!")
        print(f"Average F1 Score: {avg_f1:.3f}")
        
        return results

if __name__ == "__main__":
    validator = ModelValidator()
    results = validator.validate_on_test_set()
```

## Error Mitigation Loop

Create `scripts/error_mitigation.py`:

```python
import json
import jsonlines
from typing import List, Dict
from scripts.validate_model import ModelValidator
from scripts.prepare_dataset import DatasetPreparator

class ErrorMitigationLoop:
    def __init__(self):
        self.validator = ModelValidator()
        self.preparator = DatasetPreparator()
    
    def identify_error_patterns(self, validation_results: List[Dict]) -> Dict:
        """Analyze validation results to identify error patterns"""
        error_patterns = {
            "missing_constructs": [],
            "service_gaps": [],
            "complexity_issues": []
        }
        
        for result in validation_results:
            metrics = result.get("metrics", {})
            
            if metrics.get("f1_score", 0) < 0.5:  # Poor performance threshold
                # Analyze what went wrong
                missing = metrics.get("missing_resources", [])
                extra = metrics.get("extra_resources", [])
                
                # Check for dynamic blocks or count usage
                input_text = result.get("prediction", "")
                if "dynamic" in input_text.lower():
                    error_patterns["missing_constructs"].append("dynamic_blocks")
                if "count" in input_text.lower():
                    error_patterns["missing_constructs"].append("count")
                
                # Check for service coverage gaps
                for resource in missing:
                    service = resource.split(".")[0].replace("aws_", "")
                    error_patterns["service_gaps"].append(service)
        
        return error_patterns
    
    def generate_augmented_examples(self, error_patterns: Dict) -> List[Dict]:
        """Generate synthetic examples to address error patterns"""
        augmented_examples = []
        
        # Add more dynamic block examples
        if "dynamic_blocks" in error_patterns["missing_constructs"]:
            augmented_examples.extend(self.create_dynamic_block_examples())
        
        # Add more count examples
        if "count" in error_patterns["missing_constructs"]:
            augmented_examples.extend(self.create_count_examples())
        
        # Add examples for under-represented services
        missing_services = set(error_patterns["service_gaps"])
        for service in missing_services:
            augmented_examples.extend(self.create_service_examples(service))
        
        return augmented_examples
    
    def create_dynamic_block_examples(self) -> List[Dict]:
        """Create examples with dynamic blocks"""
        examples = []
        
        # Security group with dynamic ingress
        sg_example = {
            "input": """
# File: main.tf
variable "ingress_rules" {
  type = list(object({
    from_port   = number
    to_port     = number
    protocol    = string
    cidr_blocks = list(string)
  }))
}

resource "aws_security_group" "app" {
  name        = "app-sg"
  description = "Security group for application"
  
  dynamic "ingress" {
    for_each = var.ingress_rules
    content {
      from_port   = ingress.value.from_port
      to_port     = ingress.value.to_port
      protocol    = ingress.value.protocol
      cidr_blocks = ingress.value.cidr_blocks
    }
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# Variables: terraform.tfvars
ingress_rules = [
  {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  },
  {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  },
  {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/8"]
  }
]
""",
            "output": json.dumps([{
                "address": "aws_security_group.app",
                "type": "aws_security_group",
                "provider_name": "registry.terraform.io/hashicorp/aws",
                "change": {
                    "actions": ["create"],
                    "after": {
                        "name": "app-sg",
                        "description": "Security group for application",
                        "ingress": [
                            {
                                "from_port": 80,
                                "to_port": 80,
                                "protocol": "tcp",
                                "cidr_blocks": ["0.0.0.0/0"]
                            },
                            {
                                "from_port": 443,
                                "to_port": 443,
                                "protocol": "tcp",
                                "cidr_blocks": ["0.0.0.0/0"]
                            },
                            {
                                "from_port": 22,
                                "to_port": 22,
                                "protocol": "tcp",
                                "cidr_blocks": ["10.0.0.0/8"]
                            }
                        ],
                        "egress": [{
                            "from_port": 0,
                            "to_port": 0,
                            "protocol": "-1",
                            "cidr_blocks": ["0.0.0.0/0"]
                        }]
                    }
                }
            }]),
            "metadata": {
                "resource_count": 1,
                "aws_services": ["security_group"],
                "has_dynamic_blocks": True,
                "synthetic": True
            },
            "repository": "synthetic/dynamic_security_group"
        }
        
        examples.append(sg_example)
        return examples
    
    def create_count_examples(self) -> List[Dict]:
        """Create examples with count meta-argument"""
        examples = []
        
        # Multiple instances with count
        count_example = {
            "input": """
# File: main.tf
variable "instance_count" {
  description = "Number of instances to create"
  type        = number
  default     = 2
}

resource "aws_instance" "web" {
  count         = var.instance_count
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.micro"
  
  tags = {
    Name        = "web-server-${count.index + 1}"
    Environment = "production"
  }
}

resource "aws_eip" "web" {
  count    = var.instance_count
  instance = aws_instance.web[count.index].id
  domain   = "vpc"
  
  tags = {
    Name = "web-eip-${count.index + 1}"
  }
}

# Variables: terraform.tfvars
instance_count = 2
""",
            "output": json.dumps([
                {
                    "address": "aws_instance.web[0]",
                    "type": "aws_instance",
                    "provider_name": "registry.terraform.io/hashicorp/aws",
                    "change": {
                        "actions": ["create"],
                        "after": {
                            "ami": "ami-0c02fb55956c7d316",
                            "instance_type": "t3.micro",
                            "tags": {
                                "Name": "web-server-1",
                                "Environment": "production"
                            }
                        }
                    }
                },
                {
                    "address": "aws_instance.web[1]",
                    "type": "aws_instance",
                    "provider_name": "registry.terraform.io/hashicorp/aws",
                    "change": {
                        "actions": ["create"],
                        "after": {
                            "ami": "ami-0c02fb55956c7d316",
                            "instance_type": "t3.micro",
                            "tags": {
                                "Name": "web-server-2",
                                "Environment": "production"
                            }
                        }
                    }
                },
                {
                    "address": "aws_eip.web[0]",
                    "type": "aws_eip",
                    "provider_name": "registry.terraform.io/hashicorp/aws",
                    "change": {
                        "actions": ["create"],
                        "after": {
                            "domain": "vpc"
                        }
                    }
                },
                {
                    "address": "aws_eip.web[1]",
                    "type": "aws_eip",
                    "provider_name": "registry.terraform.io/hashicorp/aws",
                    "change": {
                        "actions": ["create"],
                        "after": {
                            "domain": "vpc"
                        }
                    }
                }
            ]),
            "metadata": {
                "resource_count": 4,
                "aws_services": ["instance", "eip"],
                "has_count": True,
                "synthetic": True
            },
            "repository": "synthetic/count_example"
        }
        
        examples.append(count_example)
        return examples
    
    def create_service_examples(self, service: str) -> List[Dict]:
        """Create examples for specific AWS services"""
        # This would contain service-specific example generation
        # For brevity, returning empty list here
        return []
    
    def run_mitigation_cycle(self, validation_results: List[Dict]) -> bool:
        """Run one cycle of error mitigation"""
        print("Analyzing error patterns...")
        error_patterns = self.identify_error_patterns(validation_results)
        
        print("Generating augmented examples...")
        augmented_examples = self.generate_augmented_examples(error_patterns)
        
        if not augmented_examples:
            print("No patterns found for augmentation")
            return False
        
        print(f"Generated {len(augmented_examples)} augmented examples")
        
        # Add to training dataset
        self.add_to_training_dataset(augmented_examples)
        
        # Re-run training (this would trigger retraining)
        print("Augmentation complete. Re-run training with updated dataset.")
        return True
    
    def add_to_training_dataset(self, new_examples: List[Dict]):
        """Add new examples to training dataset"""
        train_file = "data/processed/train.jsonl"
        
        with jsonlines.open(train_file, 'a') as writer:
            for example in new_examples:
                formatted_example = self.preparator.create_training_format(example)
                if formatted_example:
                    writer.write(formatted_example)
        
        print(f"Added {len(new_examples)} examples to training dataset")

if __name__ == "__main__":
    mitigator = ErrorMitigationLoop()
    
    # Load previous validation results
    with open("outputs/validation_results.json") as f:
        validation_results = json.load(f)
    
    # Run mitigation cycle
    improved = mitigator.run_mitigation_cycle(validation_results)
    
    if improved:
        print("Error mitigation cycle complete. Retrain the model.")
    else:
        print("No improvements identified.")
```

## Control Dataset Creation

Create `scripts/create_control_dataset.py`:

```python
import json
import tempfile
import subprocess
import os
from pathlib import Path
from typing import List, Dict

class ControlDatasetCreator:
    def __init__(self):
        self.control_repos = [
            {
                "name": "terraform-aws-vpc",
                "url": "https://github.com/terraform-aws-modules/terraform-aws-vpc.git",
                "test_dir": "examples/simple-vpc"
            },
            {
                "name": "terraform-aws-ec2-instance",
                "url": "https://github.com/terraform-aws-modules/terraform-aws-ec2-instance.git",
                "test_dir": "examples/basic"
            },
            {
                "name": "terraform-aws-rds",
                "url": "https://github.com/terraform-aws-modules/terraform-aws-rds.git",
                "test_dir": "examples/complete-mysql"
            },
            {
                "name": "terraform-aws-s3-bucket",
                "url": "https://github.com/terraform-aws-modules/terraform-aws-s3-bucket.git",
                "test_dir": "examples/complete"
            }
        ]
    
    def create_control_dataset(self):
        """Create control dataset from selected repositories"""
        control_samples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.control_repos:
                print(f"Processing control repo: {repo['name']}")
                
                repo_dir = os.path.join(temp_dir, repo["name"])
                
                # Clone repository
                try:
                    subprocess.run(
                        ["git", "clone", repo["url"], repo_dir],
                        check=True,
                        capture_output=True
                    )
                except subprocess.CalledProcessError:
                    print(f"Failed to clone {repo['name']}")
                    continue
                
                # Process test directory
                test_path = os.path.join(repo_dir, repo["test_dir"])
                if os.path.exists(test_path):
                    sample = self.process_terraform_directory(test_path, repo["name"])
                    if sample:
                        control_samples.append(sample)
        
        # Save control dataset
        os.makedirs("data/control", exist_ok=True)
        with jsonlines.open("data/control/control_dataset.jsonl", 'w') as writer:
            for sample in control_samples:
                writer.write(sample)
        
        print(f"Created control dataset with {len(control_samples)} samples")
        return control_samples
    
    def process_terraform_directory(self, tf_dir: str, repo_name: str) -> Dict:
        """Process a Terraform directory to create control sample"""
        # Collect Terraform files
        input_content = self.collect_terraform_files(tf_dir)
        
        # Run Terraform commands to get ground truth
        ground_truth = self.run_terraform_commands(tf_dir)
        
        if not ground_truth:
            return None
        
        return {
            "repository": repo_name,
            "input": input_content,
            "output": ground_truth,
            "metadata": {
                "is_control": True,
                "directory": tf_dir
            }
        }
    
    def collect_terraform_files(self, tf_dir: str) -> str:
        """Collect and concatenate Terraform files"""
        content_parts = []
        
        for tf_file in Path(tf_dir).glob("*.tf"):
            content = tf_file.read_text()
            content_parts.append(f"# File: {tf_file.name}\n{content}")
        
        for tfvars_file in Path(tf_dir).glob("*.tfvars"):
            content = tfvars_file.read_text()
            content_parts.append(f"# Variables: {tfvars_file.name}\n{content}")
        
        return "\n\n".join(content_parts)
    
    def run_terraform_commands(self, tf_dir: str) -> str:
        """Run Terraform commands to get ground truth"""
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tf_dir)
            
            # Initialize
            subprocess.run(["terraform", "init", "-backend=false"], 
                         check=True, capture_output=True)
            
            # Plan
            subprocess.run(["terraform", "plan", "-out=tfplan"], 
                         check=True, capture_output=True)
            
            # Show JSON
            result = subprocess.run(["terraform", "show", "-json", "tfplan"], 
                                  check=True, capture_output=True, text=True)
            
            plan_data = json.loads(result.stdout)
            resource_changes = []
            
            for change in plan_data.get("resource_changes", []):
                if change.get("change", {}).get("after") is not None:
                    resource_changes.append({
                        "address": change["address"],
                        "type": change["type"],
                        "provider_name": change["provider_name"],
                        "change": {
                            "actions": change["change"]["actions"],
                            "after": change["change"]["after"]
                        }
                    })
            
            return json.dumps(resource_changes, indent=2)
            
        except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
            print(f"Error running Terraform commands: {e}")
            return None
        finally:
            os.chdir(original_cwd)

if __name__ == "__main__":
    creator = ControlDatasetCreator()
    control_samples = creator.create_control_dataset()
```

## Automation Script

Create `scripts/run_complete_pipeline.py`:

```python
#!/usr/bin/env python3
"""
Complete automation pipeline for Terraform prediction model training
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command: str, description: str):
    """Run a command with error handling"""
    print(f"\n{'='*50}")
    print(f"STEP: {description}")
    print(f"COMMAND: {command}")
    print('='*50)
    
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(1)
    else:
        print(f"SUCCESS: {description} completed")
        if result.stdout:
            print(f"OUTPUT: {result.stdout}")

def main():
    """Run the complete pipeline"""
    print("Starting Terraform Prediction Model Pipeline")
    
    # Ensure we're in the correct directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Phase 1: Data Collection
    run_command(
        "python scripts/collect_terraform_repos.py",
        "Collecting Terraform repositories from GitHub"
    )
    
    # Phase 2: Ground Truth Generation
    run_command(
        "python scripts/generate_ground_truth.py",
        "Generating ground truth dataset using Terraform CLI"
    )
    
    # Phase 3: Dataset Preparation
    run_command(
        "python scripts/prepare_dataset.py",
        "Preparing training/validation/test datasets"
    )
    
    # Phase 4: Control Dataset Creation
    run_command(
        "python scripts/create_control_dataset.py",
        "Creating control dataset for validation"
    )
    
    # Phase 5: Model Fine-tuning
    run_command(
        "python scripts/fine_tune_model.py",
        "Fine-tuning Llama-7B model on Terraform dataset"
    )
    
    # Phase 6: Initial Validation
    run_command(
        "python scripts/validate_model.py",
        "Validating fine-tuned model on test dataset"
    )
    
    # Phase 7: Error Mitigation (if needed)
    run_command(
        "python scripts/error_mitigation.py",
        "Running error mitigation loop"
    )
    
    print("\n" + "="*50)
    print("PIPELINE COMPLETE!")
    print("="*50)
    print("Check the following outputs:")
    print("- models/terraform-predictor/ (fine-tuned model)")
    print("- outputs/validation_results.json (validation metrics)")
    print("- data/processed/ (prepared datasets)")
    print("- logs/ (training logs)")

if __name__ == "__main__":
    main()
```

## Remote Server Setup

### SSH Connection and Initial Setup

```bash
# Connect to remote server
ssh arnav@manthram.tplinkdns.com -p 999
# Password: HYMDLajogLLM

# Change password immediately
passwd

# Update system and install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3-pip git curl wget unzip -y

# Install Terraform
wget https://releases.hashicorp.com/terraform/1.6.0/terraform_1.6.0_linux_amd64.zip
unzip terraform_1.6.0_linux_amd64.zip
sudo mv terraform /usr/local/bin/
terraform --version

# Install Python dependencies
pip3 install torch transformers datasets accelerate bitsandbytes peft
pip3 install requests beautifulsoup4 gitpython
pip3 install zstandard jsonlines pandas numpy
pip3 install huggingface-hub wandb

# Setup CUDA (if GPU available)
nvidia-smi  # Check GPU status
```

### Transfer Project Files

```bash
# From local machine, transfer project files
scp -P 999 -r ./terraform-prediction-project arnav@manthram.tplinkdns.com:~/

# Or clone from repository
git clone <your-repo-url> terraform-prediction-project
cd terraform-prediction-project
```

## Usage Instructions

### 1. Environment Setup
```bash
# Set environment variables
export GITHUB_TOKEN=your_github_token_here
export WANDB_API_KEY=your_wandb_key_here
export HF_HOME=./huggingface_cache

# Create directory structure
mkdir -p {data,scripts,models,outputs,logs}
```

### 2. Run Complete Pipeline
```bash
# Make automation script executable
chmod +x scripts/run_complete_pipeline.py

# Run the complete pipeline
python scripts/run_complete_pipeline.py
```

### 3. Manual Step-by-Step Execution

If you prefer to run steps manually:

```bash
# Step 1: Collect repositories
python scripts/collect_terraform_repos.py

# Step 2: Generate ground truth
python scripts/generate_ground_truth.py

# Step 3: Prepare dataset
python scripts/prepare_dataset.py

# Step 4: Create control dataset
python scripts/create_control_dataset.py

# Step 5: Fine-tune model
python scripts/fine_tune_model.py

# Step 6: Validate model
python scripts/validate_model.py

# Step 7: Error mitigation (if needed)
python scripts/error_mitigation.py
```

### 4. Testing Your Model

```bash
# Test on new Terraform configuration
python -c "
from scripts.validate_model import ModelValidator
validator = ModelValidator()
config = '''
resource \"aws_s3_bucket\" \"example\" {
  bucket = \"my-test-bucket\"
}
'''
prediction = validator.predict_terraform_output(config)
print(prediction)
"
```

## Project Structure

```
terraform-prediction-project/
├── README.md
├── scripts/
│   ├── collect_terraform_repos.py      # Repository collection
│   ├── generate_ground_truth.py        # Ground truth generation
│   ├── prepare_dataset.py              # Dataset preparation
│   ├── setup_model.py                  # Model setup utilities
│   ├── fine_tune_model.py              # Fine-tuning script
│   ├── validate_model.py               # Model validation
│   ├── error_mitigation.py             # Error mitigation loop
│   ├── create_control_dataset.py       # Control dataset creation
│   └── run_complete_pipeline.py        # Complete automation
├── data/
│   ├── terraform_repos.json            # Repository metadata
│   ├── ground_truth/                   # Ground truth data
│   │   ├── complete_dataset.jsonl
│   │   └── dataset_statistics.json
│   ├── processed/                      # Processed datasets
│   │   ├── train.jsonl
│   │   ├── validation.jsonl
│   │   └── test.jsonl
│   └── control/                        # Control dataset
│       └── control_dataset.jsonl
├── models/
│   └── terraform-predictor/            # Fine-tuned model
├── outputs/
│   ├── validation_results.json         # Validation metrics
│   └── predictions/                    # Model predictions
└── logs/                               # Training logs
```

## Key Features

### 1. Comprehensive Data Collection
- Searches GitHub for diverse Terraform repositories
- Focuses on top 50 AWS services
- Collects repositories with varying complexity levels
- Analyzes resource counts and service coverage

### 2. Robust Ground Truth Generation
- Uses official Terraform CLI commands
- Handles multiple Terraform directories per repository
- Filters out resources with unknown values
- Compresses data efficiently with Zstandard

### 3. Advanced Model Training
- Uses Llama-2-7B as base model
- Implements QLoRA for efficient fine-tuning
- 4-bit quantization with NF4
- Context window optimized for large Terraform files

### 4. Intelligent Validation
- Compares predictions against Terraform CLI output
- Calculates precision, recall, and F1 scores
- Identifies specific error patterns
- Provides detailed analysis of missing/extra resources

### 5. Error Mitigation Loop
- Automatically identifies underperforming areas
- Generates synthetic examples for missing constructs
- Handles dynamic blocks and count meta-arguments
- Iteratively improves model performance

## Expected Outcomes

1. **Dataset**: 1000+ training samples from 50+ repositories
2. **Model**: Fine-tuned Llama-7B achieving >80% F1 score on validation set
3. **Validation**: Comprehensive comparison against Terraform CLI output
4. **Documentation**: Complete analysis of model performance and limitations

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in training arguments
   - Use gradient checkpointing
   - Consider using a smaller model (3B instead of 7B)

2. **Terraform Init Failures**
   - Check for required provider versions
   - Ensure network connectivity for provider downloads
   - Skip repositories with complex backend configurations

3. **Dataset Quality Issues**
   - Filter out repositories with too many unknown values
   - Focus on simpler configurations initially
   - Manually review samples for quality

### Performance Optimization

1. **Training Speed**
   - Use multiple GPUs with data parallelism
   - Optimize batch size and gradient accumulation
   - Use mixed precision training

2. **Memory Usage**
   - Enable gradient checkpointing
   - Use CPU offloading for optimizer states
   - Clear cache regularly during data processing

## Future Enhancements

1. **Multi-Provider Support**: Extend to Azure and GCP
2. **Module Resolution**: Handle Terraform module calls
3. **State Management**: Incorporate existing state into predictions
4. **Real-time Inference**: Deploy model as API service
5. **IDE Integration**: Create VSCode extension for real-time predictions

## Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Submit pull request with detailed description

## License

MIT License - see LICENSE file for details

---

**Note**: This project is for research and demonstration purposes. Always validate AI predictions against actual Terraform planning before applying infrastructure changes in production environments.
