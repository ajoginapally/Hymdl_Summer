"""
Ground truth generator using Terraform CLI
"""

import os
import json
import subprocess
import tempfile
import shutil
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
import jsonlines
import zstandard as zstd

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import (
    retry_on_failure, safe_file_operation, ProgressTracker,
    run_terraform_command, validate_terraform_directory, monitor_system_resources
)
from .terraform_analyzer import TerraformAnalyzer

logger = logging.getLogger(__name__)

class GroundTruthGenerator:
    """Generate ground truth data using official Terraform CLI"""
    
    def __init__(self):
        self.analyzer = TerraformAnalyzer()
        self.output_dir = config.data_dir / "ground_truth"
        self.output_dir.mkdir(exist_ok=True)
        
        # Verify Terraform installation
        self._verify_terraform_installation()
    
    def _verify_terraform_installation(self):
        """Verify that Terraform is properly installed"""
        try:
            result = subprocess.run(
                ["terraform", "version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                version_info = result.stdout.strip()
                logger.info(f"Terraform installation verified: {version_info}")
            else:
                raise RuntimeError("Terraform not properly installed")
                
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            logger.error(f"Terraform installation check failed: {e}")
            raise RuntimeError("Please install Terraform CLI")
    
    def load_repository_data(self, repo_file: str) -> List[Dict]:
        """Load repository data from collection output"""
        try:
            with open(repo_file, 'r') as f:
                data = json.load(f)
            
            repositories = data.get("repositories", [])
            logger.info(f"Loaded {len(repositories)} repositories from {repo_file}")
            return repositories
            
        except Exception as e:
            logger.error(f"Failed to load repository data from {repo_file}: {e}")
            return []
    
    @retry_on_failure(max_retries=2, delay=5.0)
    def clone_repository_for_processing(self, repo_data: Dict, target_dir: str) -> bool:
        """Clone repository specifically for ground truth generation"""
        try:
            import git
            
            # Clean target directory if it exists
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            
            git.Repo.clone_from(
                repo_data["clone_url"],
                target_dir,
                depth=1,
                timeout=600  # Longer timeout for stable cloning
            )
            
            logger.debug(f"Successfully cloned {repo_data['full_name']}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to clone {repo_data['full_name']}: {e}")
            return False
    
    def find_terraform_directories(self, repo_path: str) -> List[str]:
        """Find all directories containing processable Terraform files"""
        tf_directories = []
        
        for root, dirs, files in os.walk(repo_path):
            # Skip hidden directories and common non-deployment directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in [
                'node_modules', '__pycache__', '.git', '.terraform', 'dist', 'build'
            ]]
            
            if validate_terraform_directory(root):
                tf_directories.append(root)
        
        logger.debug(f"Found {len(tf_directories)} Terraform directories in {repo_path}")
        return tf_directories
    
    def run_terraform_workflow(self, tf_dir: str, provider: str = "aws") -> Optional[Dict]:
        """Run complete Terraform workflow: init, plan, show"""
        logger.debug(f"Running Terraform workflow in {tf_dir}")
        
        original_cwd = os.getcwd()
        
        try:
            os.chdir(tf_dir)
            
            # Step 1: Initialize Terraform
            logger.debug(f"Initializing Terraform in {tf_dir}")
            success, stdout, stderr = run_terraform_command(
                ["terraform", "init", "-backend=false", "-upgrade"],
                tf_dir,
                timeout=600
            )
            
            if not success:
                logger.warning(f"Terraform init failed in {tf_dir}: {stderr}")
                return None
            
            # Step 2: Validate configuration
            success, stdout, stderr = run_terraform_command(
                ["terraform", "validate"],
                tf_dir,
                timeout=120
            )
            
            if not success:
                logger.warning(f"Terraform validation failed in {tf_dir}: {stderr}")
                return None
            
            # Step 3: Create execution plan
            logger.debug(f"Creating Terraform plan in {tf_dir}")
            
            # Use different plan strategies based on provider
            plan_args = ["terraform", "plan", "-out=tfplan", "-input=false"]
            
            # Add provider-specific configurations
            if provider == "azure":
                # For Azure, we might need to set some default values
                plan_args.extend(["-var", "location=East US"])
            
            success, stdout, stderr = run_terraform_command(
                plan_args,
                tf_dir,
                timeout=900  # Longer timeout for plan
            )
            
            if not success:
                logger.warning(f"Terraform plan failed in {tf_dir}: {stderr}")
                # Try with refresh=false in case of state issues
                plan_args.append("-refresh=false")
                success, stdout, stderr = run_terraform_command(
                    plan_args,
                    tf_dir,
                    timeout=600
                )
                
                if not success:
                    logger.warning(f"Terraform plan failed again in {tf_dir}: {stderr}")
                    return None
            
            # Step 4: Export plan as JSON
            logger.debug(f"Exporting plan JSON from {tf_dir}")
            success, stdout, stderr = run_terraform_command(
                ["terraform", "show", "-json", "tfplan"],
                tf_dir,
                timeout=300
            )
            
            if not success:
                logger.warning(f"Terraform show failed in {tf_dir}: {stderr}")
                return None
            
            try:
                plan_data = json.loads(stdout)
                return plan_data
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse Terraform JSON output: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Terraform workflow failed in {tf_dir}: {e}")
            return None
        
        finally:
            os.chdir(original_cwd)
            
            # Clean up generated files
            for file_path in [os.path.join(tf_dir, f) for f in ["tfplan", "terraform.tfstate", "terraform.tfstate.backup"]]:
                if os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                    except Exception:
                        pass
    
    def extract_resource_changes(self, plan_data: Dict) -> List[Dict]:
        """Extract and filter resource changes from Terraform plan"""
        if not plan_data or "resource_changes" not in plan_data:
            return []
        
        clean_changes = []
        
        for change in plan_data["resource_changes"]:
            try:
                # Only process creates and updates with valid 'after' state
                change_actions = change.get("change", {}).get("actions", [])
                after_state = change.get("change", {}).get("after")
                
                if after_state is None:
                    continue
                
                # Skip if contains unknown values
                if self._contains_unknown_values(after_state):
                    continue
                
                # Extract clean resource change
                clean_change = {
                    "address": change["address"],
                    "type": change["type"], 
                    "name": change["name"],
                    "provider_name": change.get("provider_name", ""),
                    "change": {
                        "actions": change_actions,
                        "after": self._clean_resource_state(after_state)
                    }
                }
                
                clean_changes.append(clean_change)
                
            except Exception as e:
                logger.warning(f"Error processing resource change: {e}")
                continue
        
        return clean_changes
    
    def _contains_unknown_values(self, obj: Any, depth: int = 0) -> bool:
        """Recursively check for unknown values in nested objects"""
        if depth > 20:  # Prevent infinite recursion
            return False
        
        if isinstance(obj, dict):
            for value in obj.values():
                if self._is_unknown_value(value) or self._contains_unknown_values(value, depth + 1):
                    return True
        elif isinstance(obj, list):
            for item in obj:
                if self._is_unknown_value(item) or self._contains_unknown_values(item, depth + 1):
                    return True
        elif self._is_unknown_value(obj):
            return True
        
        return False
    
    def _is_unknown_value(self, value: Any) -> bool:
        """Check if a value represents an unknown/computed value"""
        if isinstance(value, str):
            unknown_indicators = [
                "(known after apply)",
                "(sensitive value)",
                "<computed>",
                "null_resource"
            ]
            return any(indicator in value for indicator in unknown_indicators)
        
        return False
    
    def _clean_resource_state(self, state: Dict) -> Dict:
        """Clean resource state by removing sensitive/computed values"""
        if not isinstance(state, dict):
            return state
        
        cleaned = {}
        
        for key, value in state.items():
            # Skip sensitive or computed fields
            if key in ["id", "arn", "tags_all"] or key.startswith("_"):
                continue
            
            # Recursively clean nested objects
            if isinstance(value, dict):
                cleaned_value = self._clean_resource_state(value)
                if cleaned_value:  # Only include non-empty dicts
                    cleaned[key] = cleaned_value
            elif isinstance(value, list):
                cleaned_items = []
                for item in value:
                    if isinstance(item, dict):
                        cleaned_item = self._clean_resource_state(item)
                        if cleaned_item:
                            cleaned_items.append(cleaned_item)
                    elif not self._is_unknown_value(item):
                        cleaned_items.append(item)
                
                if cleaned_items:
                    cleaned[key] = cleaned_items
            elif not self._is_unknown_value(value):
                cleaned[key] = value
        
        return cleaned
    
    def process_repository(self, repo_data: Dict) -> List[Dict]:
        """Process a single repository to generate training samples"""
        samples = []
        repo_name = repo_data["full_name"]
        
        logger.info(f"Processing repository: {repo_name}")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            repo_dir = os.path.join(temp_dir, "repo")
            
            # Clone repository
            if not self.clone_repository_for_processing(repo_data, repo_dir):
                return samples
            
            # Find Terraform directories
            tf_dirs = self.find_terraform_directories(repo_dir)
            
            if not tf_dirs:
                logger.warning(f"No valid Terraform directories found in {repo_name}")
                return samples
            
            # Determine primary provider
            primary_provider = self._detect_primary_provider(repo_data)
            
            # Process each directory (limit to avoid overwhelming)
            max_dirs = min(len(tf_dirs), config.max_samples_per_repo)
            
            for i, tf_dir in enumerate(tf_dirs[:max_dirs]):
                try:
                    logger.debug(f"Processing directory {i+1}/{max_dirs}: {tf_dir}")
                    
                    # Extract Terraform configuration
                    config_data = self.analyzer.extract_terraform_configuration(tf_dir)
                    
                    if not config_data["input_content"].strip():
                        logger.debug(f"Empty configuration in {tf_dir}")
                        continue
                    
                    # Run Terraform workflow
                    plan_data = self.run_terraform_workflow(tf_dir, primary_provider)
                    
                    if not plan_data:
                        logger.debug(f"No plan data generated for {tf_dir}")
                        continue
                    
                    # Extract resource changes
                    resource_changes = self.extract_resource_changes(plan_data)
                    
                    if not resource_changes:
                        logger.debug(f"No valid resource changes in {tf_dir}")
                        continue
                    
                    # Create training sample
                    sample = {
                        "id": f"{repo_name.replace('/', '_')}_{i}",
                        "repository": repo_name,
                        "directory": os.path.relpath(tf_dir, repo_dir),
                        "primary_provider": primary_provider,
                        "input": config_data["input_content"],
                        "output": json.dumps(resource_changes, sort_keys=True),
                        "metadata": {
                            "resource_count": len(resource_changes),
                            "aws_resources": [r["type"] for r in resource_changes if r["type"].startswith("aws_")],
                            "azure_resources": [r["type"] for r in resource_changes if r["type"].startswith("azurerm_")],
                            "complexity_score": config_data.get("analysis", {}).get("summary", {}).get("total_complexity", 0),
                            "file_count": len(config_data["file_contents"]),
                            "has_variables": len(config_data["variable_values"]) > 0,
                            "generation_timestamp": datetime.now().isoformat()
                        },
                        "terraform_output": {
                            "plan_summary": plan_data.get("planned_values", {}),
                            "configuration": plan_data.get("configuration", {}),
                            "provider_schemas": plan_data.get("provider_schemas", {})
                        }
                    }
                    
                    samples.append(sample)
                    logger.debug(f"Generated sample with {len(resource_changes)} resources")
                    
                except Exception as e:
                    logger.warning(f"Error processing directory {tf_dir}: {e}")
                    continue
        
        logger.info(f"Generated {len(samples)} samples from repository {repo_name}")
        return samples
    
    def _detect_primary_provider(self, repo_data: Dict) -> str:
        """Detect the primary cloud provider used in repository"""
        # Check repository analysis if available
        if "analysis" in repo_data:
            analysis = repo_data["analysis"]
            aws_count = len(analysis.get("aws_resources", []))
            azure_count = len(analysis.get("azure_resources", []))
            
            if azure_count > aws_count:
                return "azure"
            else:
                return "aws"
        
        # Check Azure analysis if available
        if "azure_analysis" in repo_data:
            return "azure"
        
        # Fallback to repository name/description analysis
        name_desc = f"{repo_data.get('name', '')} {repo_data.get('description', '')}".lower()
        
        if any(keyword in name_desc for keyword in ["azure", "azurerm", "microsoft"]):
            return "azure"
        
        return "aws"  # Default to AWS
    
    def process_repositories_parallel(self, repositories: List[Dict]) -> List[Dict]:
        """Process multiple repositories in parallel"""
        all_samples = []
        total_repos = len(repositories)
        
        logger.info(f"Processing {total_repos} repositories for ground truth generation")
        progress = ProgressTracker(total_repos, "Ground Truth Generation")
        
        # Process repositories in smaller batches to manage memory
        batch_size = min(config.max_workers * 2, 10)
        
        for batch_start in range(0, total_repos, batch_size):
            batch_end = min(batch_start + batch_size, total_repos)
            batch_repos = repositories[batch_start:batch_end]
            
            logger.info(f"Processing batch {batch_start//batch_size + 1}: repos {batch_start+1}-{batch_end}")
            
            # Monitor system resources
            monitor_system_resources()
            
            # Process batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
                future_to_repo = {
                    executor.submit(self.process_repository, repo): repo 
                    for repo in batch_repos
                }
                
                for future in concurrent.futures.as_completed(future_to_repo):
                    repo = future_to_repo[future]
                    try:
                        samples = future.result()
                        all_samples.extend(samples)
                        progress.update()
                        
                        # Save intermediate results periodically
                        if len(all_samples) % 50 == 0:
                            self._save_intermediate_results(all_samples, f"intermediate_{len(all_samples)}")
                            
                    except Exception as e:
                        logger.error(f"Repository processing failed for {repo.get('full_name')}: {e}")
                        progress.update()
            
            # Cleanup between batches
            import gc
            gc.collect()
        
        progress.finish()
        
        logger.info(f"Ground truth generation complete! Generated {len(all_samples)} total samples")
        return all_samples
    
    def _save_intermediate_results(self, samples: List[Dict], suffix: str):
        """Save intermediate results to prevent data loss"""
        intermediate_file = self.output_dir / f"samples_{suffix}.jsonl"
        
        try:
            with jsonlines.open(intermediate_file, 'w') as writer:
                for sample in samples:
                    writer.write(sample)
            
            logger.info(f"Saved {len(samples)} intermediate samples to {intermediate_file}")
            
        except Exception as e:
            logger.error(f"Failed to save intermediate results: {e}")
    
    def save_ground_truth_dataset(self, samples: List[Dict]):
        """Save complete ground truth dataset with compression"""
        if not samples:
            logger.warning("No samples to save")
            return
        
        # Save uncompressed JSONL
        dataset_file = self.output_dir / "complete_dataset.jsonl"
        with jsonlines.open(dataset_file, 'w') as writer:
            for sample in samples:
                writer.write(sample)
        
        # Create compressed version
        compressed_file = self.output_dir / "complete_dataset.jsonl.zst"
        with open(dataset_file, 'rb') as f_in:
            with open(compressed_file, 'wb') as f_out:
                cctx = zstd.ZstdCompressor(level=3)
                f_out.write(cctx.compress(f_in.read()))
        
        # Generate dataset statistics
        stats = self._generate_dataset_statistics(samples)
        stats_file = self.output_dir / "dataset_statistics.json"
        safe_file_operation("write", str(stats_file), stats)
        
        # Create dataset manifest
        manifest = {
            "dataset_info": {
                "total_samples": len(samples),
                "generation_date": datetime.now().isoformat(),
                "file_sizes": {
                    "uncompressed_mb": os.path.getsize(dataset_file) / (1024 * 1024),
                    "compressed_mb": os.path.getsize(compressed_file) / (1024 * 1024)
                }
            },
            "files": {
                "dataset": str(dataset_file.name),
                "compressed": str(compressed_file.name),
                "statistics": str(stats_file.name)
            }
        }
        
        manifest_file = self.output_dir / "manifest.json"
        safe_file_operation("write", str(manifest_file), manifest)
        
        logger.info(f"Ground truth dataset saved:")
        logger.info(f"  Samples: {len(samples)}")
        logger.info(f"  Uncompressed: {manifest['dataset_info']['file_sizes']['uncompressed_mb']:.1f} MB")
        logger.info(f"  Compressed: {manifest['dataset_info']['file_sizes']['compressed_mb']:.1f} MB")
    
    def _generate_dataset_statistics(self, samples: List[Dict]) -> Dict:
        """Generate comprehensive dataset statistics"""
        stats = {
            "overview": {
                "total_samples": len(samples),
                "total_repositories": len(set(s["repository"] for s in samples)),
                "generation_date": datetime.now().isoformat()
            },
            "provider_distribution": {},
            "resource_distribution": {},
            "complexity_distribution": {},
            "service_coverage": {
                "aws": {"covered": set(), "total_target": len(config.aws_services)},
                "azure": {"covered": set(), "total_target": len(config.azure_services)}
            }
        }
        
        # Analyze each sample
        for sample in samples:
            metadata = sample["metadata"]
            provider = sample["primary_provider"]
            
            # Provider distribution
            stats["provider_distribution"][provider] = stats["provider_distribution"].get(provider, 0) + 1
            
            # Resource count distribution
            resource_count = metadata["resource_count"]
            bucket = f"{resource_count//5*5}-{resource_count//5*5+4}"
            stats["resource_distribution"][bucket] = stats["resource_distribution"].get(bucket, 0) + 1
            
            # Complexity distribution
            complexity = metadata.get("complexity_score", 0)
            if complexity < 10:
                complexity_bucket = "low"
            elif complexity < 30:
                complexity_bucket = "medium"
            else:
                complexity_bucket = "high"
            stats["complexity_distribution"][complexity_bucket] = stats["complexity_distribution"].get(complexity_bucket, 0) + 1
            
            # Service coverage
            aws_resources = [r.replace("aws_", "") for r in metadata.get("aws_resources", [])]
            azure_resources = [r.replace("azurerm_", "") for r in metadata.get("azure_resources", [])]
            
            stats["service_coverage"]["aws"]["covered"].update(aws_resources)
            stats["service_coverage"]["azure"]["covered"].update(azure_resources)
        
        # Convert sets to lists and calculate percentages
        for provider in ["aws", "azure"]:
            covered_services = list(stats["service_coverage"][provider]["covered"])
            total_target = stats["service_coverage"][provider]["total_target"]
            
            stats["service_coverage"][provider] = {
                "covered_services": covered_services,
                "covered_count": len(covered_services),
                "total_target": total_target,
                "coverage_percentage": (len(covered_services) / total_target) * 100 if total_target > 0 else 0
            }
        
        return stats
    
    def generate_ground_truth_from_repos(self, repo_files: List[str]) -> str:
        """Main method to generate ground truth from repository files"""
        logger.info("Starting ground truth generation process")
        
        all_repositories = []
        
        # Load repositories from all input files
        for repo_file in repo_files:
            if os.path.exists(repo_file):
                repos = self.load_repository_data(repo_file)
                all_repositories.extend(repos)
            else:
                logger.warning(f"Repository file not found: {repo_file}")
        
        if not all_repositories:
            raise ValueError("No repositories loaded from input files")
        
        # Remove duplicates
        unique_repos = {repo["id"]: repo for repo in all_repositories}.values()
        unique_repos_list = list(unique_repos)
        
        logger.info(f"Processing {len(unique_repos_list)} unique repositories")
        
        # Process repositories
        all_samples = self.process_repositories_parallel(unique_repos_list)
        
        # Save results
        self.save_ground_truth_dataset(all_samples)
        
        return str(self.output_dir / "complete_dataset.jsonl")

def main():
    """Main execution function"""
    generator = GroundTruthGenerator()
    
    # Repository data files to process
    repo_files = [
        str(config.data_dir / "raw" / "terraform_repositories.json"),
        str(config.data_dir / "azure" / "azure_terraform_repositories.json")
    ]
    
    try:
        output_file = generator.generate_ground_truth_from_repos(repo_files)
        print(f"Ground truth generation completed successfully!")
        print(f"Dataset saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Ground truth generation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
