"""
Dataset preparation and processing for model training
"""

import os
import json
import random
import jsonlines
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
from datetime import datetime
import logging
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
import zstandard as zstd

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import safe_file_operation, ProgressTracker, monitor_system_resources

logger = logging.getLogger(__name__)

class TerraformDatasetProcessor:
    """Process Terraform datasets for training"""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        
        # Setup pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
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
            if dataset_file.endswith('.zst'):
                with open(dataset_file, 'rb') as f:
                    dctx = zstd.ZstdDecompressor()
                    decompressed_data = dctx.decompress(f.read())
                    
                # Parse JSONL from decompressed data
                for line in decompressed_data.decode('utf-8').split('\n'):
                    if line.strip():
                        samples.append(json.loads(line))
            else:
                # Regular JSONL file
                with jsonlines.open(dataset_file) as reader:
                    for sample in reader:
                        samples.append(sample)
            
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
            
            # Check token length constraints
            input_tokens = self.tokenizer.encode(formatted_input, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(output_text, add_special_tokens=False)
            
            total_length = len(input_tokens) + len(output_tokens) + 10  # Buffer for special tokens
            
            if total_length > config.max_seq_length:
                logger.debug(f"Sample {sample.get('id', 'unknown')} exceeds max length ({total_length} tokens)")
                return None
            
            # Create training sample
            training_sample = {
                "id": sample.get("id", ""),
                "input": formatted_input,
                "output": output_text,
                "repository": sample.get("repository", ""),
                "primary_provider": sample.get("primary_provider", "unknown"),
                "metadata": {
                    **sample.get("metadata", {}),
                    "input_token_length": len(input_tokens),
                    "output_token_length": len(output_tokens),
                    "total_token_length": total_length,
                    "processing_timestamp": datetime.now().isoformat()
                }
            }
            
            return training_sample
            
        except Exception as e:
            logger.warning(f"Error creating training sample: {e}")
            return None
    
    def stratify_samples_by_provider(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Stratify samples by cloud provider"""
        provider_samples = {
            "aws": [],
            "azure": [],
            "mixed": [],
            "other": []
        }
        
        for sample in samples:
            provider = sample.get("primary_provider", "unknown")
            metadata = sample.get("metadata", {})
            
            aws_count = len(metadata.get("aws_resources", []))
            azure_count = len(metadata.get("azure_resources", []))
            
            if provider == "azure" or azure_count > aws_count:
                provider_samples["azure"].append(sample)
            elif provider == "aws" or aws_count > 0:
                provider_samples["aws"].append(sample)
            elif aws_count > 0 and azure_count > 0:
                provider_samples["mixed"].append(sample)
            else:
                provider_samples["other"].append(sample)
        
        # Log distribution
        for provider, provider_samples_list in provider_samples.items():
            logger.info(f"{provider.upper()} provider: {len(provider_samples_list)} samples")
        
        return provider_samples
    
    def balance_dataset(self, stratified_samples: Dict[str, List[Dict]]) -> List[Dict]:
        """Balance dataset across providers and complexity levels"""
        balanced_samples = []
        
        # Get AWS and Azure samples
        aws_samples = stratified_samples["aws"]
        azure_samples = stratified_samples["azure"]
        mixed_samples = stratified_samples["mixed"]
        
        # Balance by taking proportional amounts
        total_samples = len(aws_samples) + len(azure_samples) + len(mixed_samples)
        
        if total_samples == 0:
            logger.warning("No samples to balance")
            return balanced_samples
        
        # Target distribution: 60% AWS, 30% Azure, 10% mixed
        target_aws = int(0.6 * total_samples)
        target_azure = int(0.3 * total_samples)
        target_mixed = int(0.1 * total_samples)
        
        # Sample from each group
        balanced_samples.extend(self._sample_with_complexity_balance(aws_samples, target_aws))
        balanced_samples.extend(self._sample_with_complexity_balance(azure_samples, target_azure))
        balanced_samples.extend(self._sample_with_complexity_balance(mixed_samples, target_mixed))
        
        # Shuffle the balanced dataset
        random.shuffle(balanced_samples)
        
        logger.info(f"Balanced dataset: {len(balanced_samples)} total samples")
        return balanced_samples
    
    def _sample_with_complexity_balance(self, samples: List[Dict], target_count: int) -> List[Dict]:
        """Sample while maintaining complexity balance"""
        if not samples or target_count <= 0:
            return []
        
        if len(samples) <= target_count:
            return samples
        
        # Sort by complexity
        samples_with_complexity = []
        for sample in samples:
            complexity = sample.get("metadata", {}).get("complexity_score", 0)
            samples_with_complexity.append((complexity, sample))
        
        samples_with_complexity.sort(key=lambda x: x[0])
        
        # Sample across complexity ranges
        selected = []
        step = len(samples_with_complexity) / target_count
        
        for i in range(target_count):
            index = int(i * step)
            if index < len(samples_with_complexity):
                selected.append(samples_with_complexity[index][1])
        
        return selected
    
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
                    writer.write(processed_sample)\n            \n            # Update dataset info\n            dataset_info[\"splits\"][split_name] = {\n                \"file\": str(split_file.name),\n                \"sample_count\": len(processed_samples),\n                \"avg_input_tokens\": np.mean([s[\"metadata\"][\"input_token_length\"] for s in processed_samples]),\n                \"avg_output_tokens\": np.mean([s[\"metadata\"][\"output_token_length\"] for s in processed_samples]),\n                \"provider_distribution\": self._get_provider_distribution(processed_samples)\n            }\n            \n            logger.info(f\"Saved {len(processed_samples)} {split_name} samples to {split_file}\")\n        \n        # Save dataset information\n        info_file = self.output_dir / \"dataset_info.json\"\n        safe_file_operation(\"write\", str(info_file), dataset_info)\n        \n        return dataset_info\n    \n    def _get_provider_distribution(self, samples: List[Dict]) -> Dict[str, int]:\n        \"\"\"Get provider distribution for a set of samples\"\"\"\n        distribution = {}\n        for sample in samples:\n            provider = sample.get(\"primary_provider\", \"unknown\")\n            distribution[provider] = distribution.get(provider, 0) + 1\n        return distribution\n    \n    def generate_synthetic_samples(self) -> List[Dict]:\n        \"\"\"Generate synthetic training samples for edge cases\"\"\"\n        synthetic_samples = []\n        \n        # AWS dynamic block example\n        aws_dynamic_sample = {\n            \"id\": \"synthetic_aws_dynamic_001\",\n            \"repository\": \"synthetic/aws_examples\",\n            \"primary_provider\": \"aws\",\n            \"input\": \"\"\"# File: main.tf\nvariable \"ingress_rules\" {\n  type = list(object({\n    from_port   = number\n    to_port     = number\n    protocol    = string\n    cidr_blocks = list(string)\n  }))\n  default = [\n    {\n      from_port   = 80\n      to_port     = 80\n      protocol    = \"tcp\"\n      cidr_blocks = [\"0.0.0.0/0\"]\n    },\n    {\n      from_port   = 443\n      to_port     = 443\n      protocol    = \"tcp\"\n      cidr_blocks = [\"0.0.0.0/0\"]\n    }\n  ]\n}\n\nresource \"aws_security_group\" \"web\" {\n  name        = \"web-sg\"\n  description = \"Security group for web servers\"\n  \n  dynamic \"ingress\" {\n    for_each = var.ingress_rules\n    content {\n      from_port   = ingress.value.from_port\n      to_port     = ingress.value.to_port\n      protocol    = ingress.value.protocol\n      cidr_blocks = ingress.value.cidr_blocks\n    }\n  }\n  \n  egress {\n    from_port   = 0\n    to_port     = 0\n    protocol    = \"-1\"\n    cidr_blocks = [\"0.0.0.0/0\"]\n  }\n  \n  tags = {\n    Name = \"web-security-group\"\n  }\n}\"\"\",\n            \"output\": json.dumps([{\n                \"address\": \"aws_security_group.web\",\n                \"type\": \"aws_security_group\",\n                \"name\": \"web\",\n                \"provider_name\": \"registry.terraform.io/hashicorp/aws\",\n                \"change\": {\n                    \"actions\": [\"create\"],\n                    \"after\": {\n                        \"name\": \"web-sg\",\n                        \"description\": \"Security group for web servers\",\n                        \"ingress\": [\n                            {\n                                \"from_port\": 80,\n                                \"to_port\": 80,\n                                \"protocol\": \"tcp\",\n                                \"cidr_blocks\": [\"0.0.0.0/0\"]\n                            },\n                            {\n                                \"from_port\": 443,\n                                \"to_port\": 443,\n                                \"protocol\": \"tcp\",\n                                \"cidr_blocks\": [\"0.0.0.0/0\"]\n                            }\n                        ],\n                        \"egress\": [{\n                            \"from_port\": 0,\n                            \"to_port\": 0,\n                            \"protocol\": \"-1\",\n                            \"cidr_blocks\": [\"0.0.0.0/0\"]\n                        }],\n                        \"tags\": {\n                            \"Name\": \"web-security-group\"\n                        }\n                    }\n                }\n            }]),\n            \"metadata\": {\n                \"resource_count\": 1,\n                \"aws_resources\": [\"aws_security_group\"],\n                \"azure_resources\": [],\n                \"complexity_score\": 15,\n                \"has_dynamic_blocks\": True,\n                \"synthetic\": True\n            }\n        }\n        \n        # Azure resource group and VM example\n        azure_vm_sample = {\n            \"id\": \"synthetic_azure_vm_001\",\n            \"repository\": \"synthetic/azure_examples\",\n            \"primary_provider\": \"azure\",\n            \"input\": \"\"\"# File: main.tf\nresource \"azurerm_resource_group\" \"main\" {\n  name     = \"example-resources\"\n  location = \"East US\"\n}\n\nresource \"azurerm_virtual_network\" \"main\" {\n  name                = \"example-vnet\"\n  address_space       = [\"10.0.0.0/16\"]\n  location            = azurerm_resource_group.main.location\n  resource_group_name = azurerm_resource_group.main.name\n}\n\nresource \"azurerm_subnet\" \"internal\" {\n  name                 = \"internal\"\n  resource_group_name  = azurerm_resource_group.main.name\n  virtual_network_name = azurerm_virtual_network.main.name\n  address_prefixes     = [\"10.0.2.0/24\"]\n}\n\nresource \"azurerm_network_interface\" \"main\" {\n  name                = \"example-nic\"\n  location            = azurerm_resource_group.main.location\n  resource_group_name = azurerm_resource_group.main.name\n  \n  ip_configuration {\n    name                          = \"internal\"\n    subnet_id                     = azurerm_subnet.internal.id\n    private_ip_address_allocation = \"Dynamic\"\n  }\n}\n\nresource \"azurerm_virtual_machine\" \"main\" {\n  name                = \"example-vm\"\n  location            = azurerm_resource_group.main.location\n  resource_group_name = azurerm_resource_group.main.name\n  network_interface_ids = [\n    azurerm_network_interface.main.id,\n  ]\n  vm_size = \"Standard_DS1_v2\"\n  \n  storage_os_disk {\n    name              = \"myosdisk1\"\n    caching           = \"ReadWrite\"\n    create_option     = \"FromImage\"\n    managed_disk_type = \"Standard_LRS\"\n  }\n  \n  storage_image_reference {\n    publisher = \"Canonical\"\n    offer     = \"UbuntuServer\"\n    sku       = \"18.04-LTS\"\n    version   = \"latest\"\n  }\n  \n  os_profile {\n    computer_name  = \"hostname\"\n    admin_username = \"testadmin\"\n    admin_password = \"Password1234!\"\n  }\n  \n  os_profile_linux_config {\n    disable_password_authentication = false\n  }\n}\"\"\",\n            \"output\": json.dumps([\n                {\n                    \"address\": \"azurerm_resource_group.main\",\n                    \"type\": \"azurerm_resource_group\",\n                    \"name\": \"main\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/azurerm\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"name\": \"example-resources\",\n                            \"location\": \"East US\"\n                        }\n                    }\n                },\n                {\n                    \"address\": \"azurerm_virtual_network.main\",\n                    \"type\": \"azurerm_virtual_network\",\n                    \"name\": \"main\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/azurerm\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"name\": \"example-vnet\",\n                            \"address_space\": [\"10.0.0.0/16\"],\n                            \"location\": \"East US\",\n                            \"resource_group_name\": \"example-resources\"\n                        }\n                    }\n                },\n                {\n                    \"address\": \"azurerm_subnet.internal\",\n                    \"type\": \"azurerm_subnet\",\n                    \"name\": \"internal\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/azurerm\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"name\": \"internal\",\n                            \"resource_group_name\": \"example-resources\",\n                            \"virtual_network_name\": \"example-vnet\",\n                            \"address_prefixes\": [\"10.0.2.0/24\"]\n                        }\n                    }\n                },\n                {\n                    \"address\": \"azurerm_network_interface.main\",\n                    \"type\": \"azurerm_network_interface\",\n                    \"name\": \"main\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/azurerm\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"name\": \"example-nic\",\n                            \"location\": \"East US\",\n                            \"resource_group_name\": \"example-resources\",\n                            \"ip_configuration\": [{\n                                \"name\": \"internal\",\n                                \"private_ip_address_allocation\": \"Dynamic\"\n                            }]\n                        }\n                    }\n                },\n                {\n                    \"address\": \"azurerm_virtual_machine.main\",\n                    \"type\": \"azurerm_virtual_machine\",\n                    \"name\": \"main\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/azurerm\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"name\": \"example-vm\",\n                            \"location\": \"East US\",\n                            \"resource_group_name\": \"example-resources\",\n                            \"vm_size\": \"Standard_DS1_v2\",\n                            \"storage_os_disk\": {\n                                \"name\": \"myosdisk1\",\n                                \"caching\": \"ReadWrite\",\n                                \"create_option\": \"FromImage\",\n                                \"managed_disk_type\": \"Standard_LRS\"\n                            },\n                            \"storage_image_reference\": {\n                                \"publisher\": \"Canonical\",\n                                \"offer\": \"UbuntuServer\",\n                                \"sku\": \"18.04-LTS\",\n                                \"version\": \"latest\"\n                            },\n                            \"os_profile\": {\n                                \"computer_name\": \"hostname\",\n                                \"admin_username\": \"testadmin\"\n                            },\n                            \"os_profile_linux_config\": {\n                                \"disable_password_authentication\": False\n                            }\n                        }\n                    }\n                }\n            ]),\n            \"metadata\": {\n                \"resource_count\": 5,\n                \"aws_resources\": [],\n                \"azure_resources\": [\n                    \"azurerm_resource_group\", \"azurerm_virtual_network\", \n                    \"azurerm_subnet\", \"azurerm_network_interface\", \"azurerm_virtual_machine\"\n                ],\n                \"complexity_score\": 25,\n                \"has_dependencies\": True,\n                \"synthetic\": True\n            }\n        }\n        \n        # AWS count example\n        aws_count_sample = {\n            \"id\": \"synthetic_aws_count_001\",\n            \"repository\": \"synthetic/aws_examples\",\n            \"primary_provider\": \"aws\",\n            \"input\": \"\"\"# File: main.tf\nvariable \"instance_count\" {\n  description = \"Number of EC2 instances\"\n  type        = number\n  default     = 3\n}\n\nresource \"aws_instance\" \"web\" {\n  count         = var.instance_count\n  ami           = \"ami-0c02fb55956c7d316\"\n  instance_type = \"t3.micro\"\n  \n  tags = {\n    Name = \"web-server-${count.index + 1}\"\n    Environment = \"production\"\n  }\n}\n\nresource \"aws_eip\" \"web\" {\n  count    = var.instance_count\n  instance = aws_instance.web[count.index].id\n  domain   = \"vpc\"\n  \n  tags = {\n    Name = \"web-eip-${count.index + 1}\"\n  }\n}\"\"\",\n            \"output\": json.dumps([\n                {\n                    \"address\": \"aws_instance.web[0]\",\n                    \"type\": \"aws_instance\",\n                    \"name\": \"web\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/aws\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"ami\": \"ami-0c02fb55956c7d316\",\n                            \"instance_type\": \"t3.micro\",\n                            \"tags\": {\n                                \"Name\": \"web-server-1\",\n                                \"Environment\": \"production\"\n                            }\n                        }\n                    }\n                },\n                {\n                    \"address\": \"aws_instance.web[1]\",\n                    \"type\": \"aws_instance\",\n                    \"name\": \"web\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/aws\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"ami\": \"ami-0c02fb55956c7d316\",\n                            \"instance_type\": \"t3.micro\",\n                            \"tags\": {\n                                \"Name\": \"web-server-2\",\n                                \"Environment\": \"production\"\n                            }\n                        }\n                    }\n                },\n                {\n                    \"address\": \"aws_instance.web[2]\",\n                    \"type\": \"aws_instance\",\n                    \"name\": \"web\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/aws\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"ami\": \"ami-0c02fb55956c7d316\",\n                            \"instance_type\": \"t3.micro\",\n                            \"tags\": {\n                                \"Name\": \"web-server-3\",\n                                \"Environment\": \"production\"\n                            }\n                        }\n                    }\n                },\n                {\n                    \"address\": \"aws_eip.web[0]\",\n                    \"type\": \"aws_eip\",\n                    \"name\": \"web\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/aws\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"domain\": \"vpc\"\n                        }\n                    }\n                },\n                {\n                    \"address\": \"aws_eip.web[1]\",\n                    \"type\": \"aws_eip\",\n                    \"name\": \"web\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/aws\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"domain\": \"vpc\"\n                        }\n                    }\n                },\n                {\n                    \"address\": \"aws_eip.web[2]\",\n                    \"type\": \"aws_eip\",\n                    \"name\": \"web\",\n                    \"provider_name\": \"registry.terraform.io/hashicorp/aws\",\n                    \"change\": {\n                        \"actions\": [\"create\"],\n                        \"after\": {\n                            \"domain\": \"vpc\"\n                        }\n                    }\n                }\n            ]),\n            \"metadata\": {\n                \"resource_count\": 6,\n                \"aws_resources\": [\"aws_instance\", \"aws_eip\"],\n                \"azure_resources\": [],\n                \"complexity_score\": 20,\n                \"has_count\": True,\n                \"synthetic\": True\n            }\n        }\n        \n        synthetic_samples.extend([aws_dynamic_sample, azure_vm_sample, aws_count_sample])\n        \n        logger.info(f\"Generated {len(synthetic_samples)} synthetic samples\")\n        return synthetic_samples\n    \n    def process_complete_dataset(self, ground_truth_file: str) -> Dict[str, str]:\n        \"\"\"Main method to process complete dataset\"\"\"\n        logger.info(\"Starting dataset processing\")\n        \n        # Load ground truth samples\n        samples = self.load_ground_truth_samples(ground_truth_file)\n        \n        if not samples:\n            raise ValueError(f\"No samples loaded from {ground_truth_file}\")\n        \n        # Add synthetic samples\n        synthetic_samples = self.generate_synthetic_samples()\n        samples.extend(synthetic_samples)\n        \n        logger.info(f\"Total samples (including synthetic): {len(samples)}\")\n        \n        # Stratify by provider\n        stratified_samples = self.stratify_samples_by_provider(samples)\n        \n        # Balance dataset\n        balanced_samples = self.balance_dataset(stratified_samples)\n        \n        # Create train/validation/test splits\n        train_samples, val_samples, test_samples = self.create_train_val_test_split(balanced_samples)\n        \n        # Save dataset splits\n        dataset_info = self.save_dataset_splits(train_samples, val_samples, test_samples)\n        \n        # Create dataset summary\n        self._create_dataset_summary(dataset_info)\n        \n        output_files = {\n            \"train\": str(self.output_dir / \"train.jsonl\"),\n            \"validation\": str(self.output_dir / \"validation.jsonl\"),\n            \"test\": str(self.output_dir / \"test.jsonl\"),\n            \"info\": str(self.output_dir / \"dataset_info.json\")\n        }\n        \n        logger.info(\"Dataset processing completed successfully\")\n        return output_files\n    \n    def _create_dataset_summary(self, dataset_info: Dict):\n        \"\"\"Create human-readable dataset summary\"\"\"\n        summary = {\n            \"dataset_overview\": {\n                \"creation_date\": dataset_info[\"creation_date\"],\n                \"model_target\": dataset_info[\"model_name\"],\n                \"max_sequence_length\": dataset_info[\"max_seq_length\"]\n            },\n            \"split_summary\": {},\n            \"recommendations\": []\n        }\n        \n        total_samples = 0\n        for split_name, split_info in dataset_info[\"splits\"].items():\n            sample_count = split_info[\"sample_count\"]\n            total_samples += sample_count\n            \n            summary[\"split_summary\"][split_name] = {\n                \"samples\": sample_count,\n                \"avg_input_tokens\": round(split_info[\"avg_input_tokens\"], 1),\n                \"avg_output_tokens\": round(split_info[\"avg_output_tokens\"], 1),\n                \"provider_distribution\": split_info[\"provider_distribution\"]\n            }\n        \n        summary[\"dataset_overview\"][\"total_samples\"] = total_samples\n        \n        # Add recommendations\n        if total_samples < 500:\n            summary[\"recommendations\"].append(\n                \"Consider collecting more repositories to improve model performance\"\n            )\n        \n        train_samples = dataset_info[\"splits\"].get(\"train\", {}).get(\"sample_count\", 0)\n        if train_samples < 300:\n            summary[\"recommendations\"].append(\n                \"Training set is small. Consider data augmentation or collecting more samples\"\n            )\n        \n        # Save summary\n        summary_file = self.output_dir / \"dataset_summary.json\"\n        safe_file_operation(\"write\", str(summary_file), summary)\n        \n        logger.info(f\"Dataset summary saved to {summary_file}\")\n\ndef main():\n    \"\"\"Main execution function\"\"\"\n    processor = TerraformDatasetProcessor()\n    \n    # Input ground truth file\n    ground_truth_file = str(config.data_dir / \"ground_truth\" / \"complete_dataset.jsonl\")\n    \n    # Check if compressed version exists\n    if os.path.exists(ground_truth_file + \".zst\"):\n        ground_truth_file = ground_truth_file + \".zst\"\n    \n    if not os.path.exists(ground_truth_file):\n        logger.error(f\"Ground truth file not found: {ground_truth_file}\")\n        print(\"Please run ground truth generation first\")\n        sys.exit(1)\n    \n    try:\n        output_files = processor.process_complete_dataset(ground_truth_file)\n        \n        print(\"Dataset processing completed successfully!\")\n        print(\"Output files:\")\n        for split, filepath in output_files.items():\n            print(f\"  {split}: {filepath}\")\n        \n    except Exception as e:\n        logger.error(f\"Dataset processing failed: {e}\")\n        sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()
