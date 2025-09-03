"""
Dataset preparation and processing for model training
"""

import os
import json
import random
import jsonlines
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import logging
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import gzip

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import safe_file_operation, ProgressTracker

logger = logging.getLogger(__name__)

class TerraformDatasetProcessor:
    """Process Terraform datasets for training"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.model_name
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Setup pad token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        except Exception as e:
            logger.warning(f"Could not load tokenizer: {e}")
            self.tokenizer = None
        
        self.output_dir = config.data_dir / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
        # Training format template
        self.instruction_template = """You are a Terraform infrastructure prediction expert. Given Terraform configuration files, predict the exact resource changes that would result from running 'terraform plan'.

Respond with a JSON array containing the resource changes. Each resource should include:
- address: The Terraform resource address
- type: The resource type (e.g., 'aws_instance', 'azurerm_virtual_machine') 
- name: The resource name
- provider_name: The provider (e.g., 'registry.terraform.io/hashicorp/aws')
- change: Object with 'actions' and 'after' state

Input Terraform Configuration:
{input}

Output:"""
    
    def load_ground_truth_samples(self, dataset_file: str) -> List[Dict]:
        """Load samples from ground truth dataset"""
        samples = []
        
        try:
            # Handle compressed files
            if dataset_file.endswith('.gz'):
                with gzip.open(dataset_file, 'rt') as f:
                    data = json.load(f)
                    samples = data if isinstance(data, list) else [data]
            elif dataset_file.endswith('.jsonl'):
                with jsonlines.open(dataset_file) as reader:
                    for sample in reader:
                        samples.append(sample)
            else:
                # Regular JSON file
                with open(dataset_file) as f:
                    data = json.load(f)
                    samples = data if isinstance(data, list) else [data]
            
            logger.info(f"Loaded {len(samples)} samples from {dataset_file}")
            return samples
            
        except Exception as e:
            logger.error(f"Failed to load samples from {dataset_file}: {e}")
            return []
    
    def create_training_sample(self, sample: Dict) -> Optional[Dict]:
        """Convert ground truth sample to training format"""
        try:
            # Format instruction with input
            formatted_input = self.instruction_template.format(input=sample["input"])
            output_text = sample["output"]
            
            # Check token length constraints if tokenizer available
            if self.tokenizer:
                input_tokens = self.tokenizer.encode(formatted_input, add_special_tokens=False)
                output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
                
                total_length = len(input_tokens) + len(output_tokens) + 10  # Buffer for special tokens
                
                if total_length > config.max_seq_length:
                    logger.debug(f"Sample {sample.get('id', 'unknown')} exceeds max length ({total_length} tokens)")
                    return None
                
                input_token_length = len(input_tokens)
                output_token_length = len(output_tokens)
            else:
                # Fallback without tokenizer
                input_token_length = len(formatted_input.split())
                output_token_length = len(output_text.split())
                total_length = input_token_length + output_token_length
            
            # Create training sample
            training_sample = {
                "id": sample.get("id", ""),
                "input": formatted_input,
                "output": output_text,
                "repository": sample.get("repository", ""),
                "primary_provider": sample.get("primary_provider", "unknown"),
                "metadata": {
                    **sample.get("metadata", {}),
                    "input_token_length": input_token_length,
                    "output_token_length": output_token_length,
                    "total_token_length": total_length,
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
            return training_sample
            
        except Exception as e:
            logger.warning(f"Error creating training sample: {e}")
            return None
    
    def process_complete_dataset(self, ground_truth_file: str) -> Dict[str, str]:
        """Main method to process complete dataset"""
        logger.info("Starting dataset processing")
        
        # Load ground truth samples
        samples = self.load_ground_truth_samples(ground_truth_file)
        
        if not samples:
            raise ValueError(f"No samples loaded from {ground_truth_file}")
        
        # Add synthetic samples
        synthetic_samples = self.generate_synthetic_samples()
        samples.extend(synthetic_samples)
        
        logger.info(f"Total samples (including synthetic): {len(samples)}")
        
        # Create train/validation/test splits
        train_samples, val_samples, test_samples = self.create_train_val_test_split(samples)
        
        # Save dataset splits
        dataset_info = self.save_dataset_splits(train_samples, val_samples, test_samples)
        
        output_files = {
            "train": str(self.output_dir / "train.jsonl"),
            "validation": str(self.output_dir / "validation.jsonl"),
            "test": str(self.output_dir / "test.jsonl"),
            "info": str(self.output_dir / "dataset_info.json")
        }
        
        logger.info("Dataset processing completed successfully")
        return output_files
    
    def create_train_val_test_split(self, samples: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Create train/validation/test split by repository boundary"""
        # Group samples by repository
        repo_groups = {}
        for sample in samples:
            repo = sample.get("repository", "unknown")
            if repo not in repo_groups:
                repo_groups[repo] = []
            repo_groups[repo].append(sample)
        
        # Split repositories (not samples) to avoid data leakage
        repositories = list(repo_groups.keys())
        random.shuffle(repositories)
        
        # Calculate split points
        total_repos = len(repositories)
        train_repo_count = int(0.8 * total_repos)
        val_repo_count = int(0.1 * total_repos)
        
        # Split repositories
        train_repos = repositories[:train_repo_count]
        val_repos = repositories[train_repo_count:train_repo_count + val_repo_count]
        test_repos = repositories[train_repo_count + val_repo_count:]
        
        # Collect samples
        train_samples = []
        val_samples = []
        test_samples = []
        
        for repo in train_repos:
            train_samples.extend(repo_groups[repo])
        
        for repo in val_repos:
            val_samples.extend(repo_groups[repo])
        
        for repo in test_repos:
            test_samples.extend(repo_groups[repo])
        
        # Log split information
        logger.info(f"Dataset split:")
        logger.info(f"  Train: {len(train_samples)} samples from {len(train_repos)} repositories")
        logger.info(f"  Validation: {len(val_samples)} samples from {len(val_repos)} repositories")
        logger.info(f"  Test: {len(test_samples)} samples from {len(test_repos)} repositories")
        
        return train_samples, val_samples, test_samples
    
    def save_dataset_splits(self, train_samples: List[Dict], val_samples: List[Dict], test_samples: List[Dict]):
        """Save dataset splits to files"""
        splits = {
            "train": train_samples,
            "validation": val_samples,
            "test": test_samples
        }
        
        dataset_info = {
            "creation_date": datetime.now().isoformat(),
            "model_name": self.model_name,
            "max_seq_length": config.max_seq_length,
            "splits": {}
        }
        
        for split_name, samples in splits.items():
            if not samples:
                logger.warning(f"No samples for {split_name} split")
                continue
            
            # Convert samples to training format
            processed_samples = []
            progress = ProgressTracker(len(samples), f"Processing {split_name} samples")
            
            for sample in samples:
                training_sample = self.create_training_sample(sample)
                if training_sample:
                    processed_samples.append(training_sample)
                progress.update()
            
            progress.finish()
            
            # Save to file
            split_file = self.output_dir / f"{split_name}.jsonl"
            with jsonlines.open(split_file, 'w') as writer:
                for processed_sample in processed_samples:
                    writer.write(processed_sample)
            
            # Update dataset info
            if processed_samples:
                dataset_info["splits"][split_name] = {
                    "file": str(split_file.name),
                    "sample_count": len(processed_samples),
                    "avg_input_tokens": np.mean([s["metadata"]["input_token_length"] for s in processed_samples]),
                    "avg_output_tokens": np.mean([s["metadata"]["output_token_length"] for s in processed_samples]),
                    "provider_distribution": self._get_provider_distribution(processed_samples)
                }
            
            logger.info(f"Saved {len(processed_samples)} {split_name} samples to {split_file}")
        
        # Save dataset information
        info_file = self.output_dir / "dataset_info.json"
        safe_file_operation("write", str(info_file), dataset_info)
        
        return dataset_info
    
    def _get_provider_distribution(self, samples: List[Dict]) -> Dict[str, int]:
        """Get provider distribution for a set of samples"""
        distribution = {}
        for sample in samples:
            provider = sample.get("primary_provider", "unknown")
            distribution[provider] = distribution.get(provider, 0) + 1
        return distribution
    
    def generate_synthetic_samples(self) -> List[Dict]:
        """Generate synthetic training samples for edge cases"""
        synthetic_samples = []
        
        # Simple AWS S3 example
        aws_s3_sample = {
            "id": "synthetic_aws_s3_001",
            "repository": "synthetic/aws_examples",
            "primary_provider": "aws",
            "input": """# File: main.tf
resource "aws_s3_bucket" "example" {
  bucket = "example-bucket"
  
  tags = {
    Name = "example-bucket"
    Environment = "production"
  }
}""",
            "output": json.dumps([{
                "address": "aws_s3_bucket.example",
                "type": "aws_s3_bucket",
                "name": "example",
                "provider_name": "registry.terraform.io/hashicorp/aws",
                "change": {
                    "actions": ["create"],
                    "after": {
                        "bucket": "example-bucket",
                        "tags": {
                            "Name": "example-bucket",
                            "Environment": "production"
                        }
                    }
                }
            }]),
            "metadata": {
                "resource_count": 1,
                "aws_resources": ["aws_s3_bucket"],
                "azure_resources": [],
                "complexity_score": 5,
                "synthetic": True
            }
        }
        
        # Simple Azure Resource Group example
        azure_rg_sample = {
            "id": "synthetic_azure_rg_001",
            "repository": "synthetic/azure_examples",
            "primary_provider": "azure",
            "input": """# File: main.tf
resource "azurerm_resource_group" "example" {
  name     = "example-resources"
  location = "East US"
}""",
            "output": json.dumps([{
                "address": "azurerm_resource_group.example",
                "type": "azurerm_resource_group", 
                "name": "example",
                "provider_name": "registry.terraform.io/hashicorp/azurerm",
                "change": {
                    "actions": ["create"],
                    "after": {
                        "name": "example-resources",
                        "location": "East US"
                    }
                }
            }]),
            "metadata": {
                "resource_count": 1,
                "aws_resources": [],
                "azure_resources": ["azurerm_resource_group"],
                "complexity_score": 3,
                "synthetic": True
            }
        }
        
        synthetic_samples.extend([aws_s3_sample, azure_rg_sample])
        
        logger.info(f"Generated {len(synthetic_samples)} synthetic samples")
        return synthetic_samples

def main():
    """Main execution function"""
    if len(sys.argv) > 1:
        ground_truth_file = sys.argv[1]
    else:
        ground_truth_file = str(config.data_dir / "ground_truth" / "terraform_dataset.json.gz")
    
    if not os.path.exists(ground_truth_file):
        logger.error(f"Ground truth file not found: {ground_truth_file}")
        print("Please run ground truth generation first")
        sys.exit(1)
    
    try:
        processor = TerraformDatasetProcessor()
        output_files = processor.process_complete_dataset(ground_truth_file)
        
        print("Dataset processing completed successfully!")
        print("Output files:")
        for split, filepath in output_files.items():
            print(f"  {split}: {filepath}")
        
    except Exception as e:
        logger.error(f"Dataset processing failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
