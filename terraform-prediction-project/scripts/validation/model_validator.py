"""
Model validation and testing framework
"""

import os
import json
import torch
import tempfile
import jsonlines
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import subprocess
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import difflib

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import (
    safe_file_operation, run_terraform_command, 
    normalize_json_output, calculate_similarity
)
from ..data_collection.terraform_analyzer import TerraformAnalyzer

logger = logging.getLogger(__name__)

class TerraformModelValidator:
    """Comprehensive validation framework for Terraform prediction model"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or str(config.model_output_dir)
        self.analyzer = TerraformAnalyzer()
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Output directory for validation results
        self.output_dir = config.outputs_dir / "validation"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_trained_model(self):
        """Load the fine-tuned model and tokenizer"""
        logger.info(f"Loading trained model from {self.model_path}")
        
        try:
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                config.base_model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info("Model loaded successfully for validation")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict_terraform_output(self, terraform_config: str, max_new_tokens: int = 4096) -> str:
        """Generate prediction for Terraform configuration"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_trained_model() first")
        
        # Format input with instruction template
        instruction = """You are a Terraform infrastructure prediction expert. Given Terraform configuration files, predict the exact resource changes that would result from running 'terraform plan'.

Respond with a JSON array containing the resource changes. Each resource should include:
- address: The Terraform resource address
- type: The resource type (e.g., 'aws_instance', 'azurerm_virtual_machine')
- name: The resource name  
- provider_name: The provider (e.g., 'registry.terraform.io/hashicorp/aws')
- change: Object with 'actions' and 'after' state

Input Terraform Configuration:
{config}

Output:""".format(config=terraform_config)
        
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                instruction, 
                return_tensors="pt",
                truncation=True,
                max_length=config.max_seq_length - max_new_tokens
            ).to(self.device)
            
            # Generate prediction
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0,
                    top_p=0.9
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][len(inputs[0]):], 
                skip_special_tokens=True
            )
            
            # Extract and validate JSON
            return self._extract_and_validate_json(generated_text)
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return json.dumps({"error": str(e)})
    
    def _extract_and_validate_json(self, generated_text: str) -> str:
        """Extract and validate JSON from generated text"""
        try:
            # Try to find JSON array in the output
            text = generated_text.strip()
            
            # Look for JSON array patterns
            start_patterns = ['[', '[\n', '[\r\n']
            end_patterns = [']', '\n]', '\r\n]']
            
            json_start = -1
            json_end = -1
            
            for pattern in start_patterns:
                idx = text.find(pattern)
                if idx != -1:
                    json_start = idx
                    break
            
            if json_start != -1:
                for pattern in end_patterns:
                    idx = text.rfind(pattern)
                    if idx > json_start:
                        json_end = idx + len(pattern.rstrip('\n\r'))
                        break
            
            if json_start != -1 and json_end != -1:
                json_text = text[json_start:json_end]
                
                # Validate JSON
                parsed = json.loads(json_text)
                return json.dumps(parsed, indent=2, sort_keys=True)
            else:
                # Fallback: try to parse entire text
                parsed = json.loads(text)
                return json.dumps(parsed, indent=2, sort_keys=True)
                
        except json.JSONDecodeError:
            # If JSON parsing fails, return original text
            logger.warning("Generated output is not valid JSON")
            return generated_text
    
    def generate_ground_truth(self, terraform_config: str) -> str:
        """Generate ground truth using Terraform CLI"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Parse the configuration to extract files
                config_lines = terraform_config.split('\n')
                current_file = None
                current_content = []
                
                for line in config_lines:
                    if line.strip().startswith("# File:"):
                        # Save previous file
                        if current_file and current_content:
                            file_path = os.path.join(temp_dir, current_file)
                            with open(file_path, 'w') as f:
                                f.write('\n'.join(current_content))
                        
                        # Start new file
                        current_file = line.replace("# File:", "").strip()
                        current_content = []
                        
                    elif line.strip().startswith("# Variables:"):
                        # Save previous file
                        if current_file and current_content:
                            file_path = os.path.join(temp_dir, current_file)
                            with open(file_path, 'w') as f:
                                f.write('\n'.join(current_content))
                        
                        # Start variables file
                        current_file = line.replace("# Variables:", "").strip()
                        current_content = []
                        
                    else:
                        current_content.append(line)
                
                # Save final file
                if current_file and current_content:
                    file_path = os.path.join(temp_dir, current_file)
                    with open(file_path, 'w') as f:
                        f.write('\n'.join(current_content))
                
                # Run Terraform workflow
                success, stdout, stderr = run_terraform_command(
                    ["terraform", "init", "-backend=false"],
                    temp_dir
                )
                
                if not success:
                    return json.dumps({"error": f"Terraform init failed: {stderr}"})
                
                success, stdout, stderr = run_terraform_command(
                    ["terraform", "plan", "-out=tfplan"],
                    temp_dir,
                    timeout=900
                )
                
                if not success:
                    return json.dumps({"error": f"Terraform plan failed: {stderr}"})
                
                success, stdout, stderr = run_terraform_command(
                    ["terraform", "show", "-json", "tfplan"],
                    temp_dir
                )
                
                if not success:
                    return json.dumps({"error": f"Terraform show failed: {stderr}"})
                
                # Parse and extract resource changes
                plan_data = json.loads(stdout)
                resource_changes = []
                
                for change in plan_data.get("resource_changes", []):
                    if change.get("change", {}).get("after") is not None:
                        resource_changes.append({
                            "address": change["address"],
                            "type": change["type"],
                            "name": change["name"],
                            "provider_name": change.get("provider_name", ""),
                            "change": {
                                "actions": change["change"]["actions"],
                                "after": change["change"]["after"]
                            }
                        })
                
                return json.dumps(resource_changes, indent=2, sort_keys=True)
                
            except Exception as e:
                logger.error(f"Ground truth generation failed: {e}")
                return json.dumps({"error": str(e)})
    
    def compare_predictions(self, predicted: str, ground_truth: str) -> Dict[str, Any]:
        """Compare model prediction with ground truth"""
        try:
            pred_data = json.loads(predicted)
            truth_data = json.loads(ground_truth)
            
            # Handle error cases
            if isinstance(pred_data, dict) and "error" in pred_data:
                return {
                    "prediction_error": pred_data["error"],
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "accuracy": 0.0
                }
            
            if isinstance(truth_data, dict) and "error" in truth_data:
                return {
                    "ground_truth_error": truth_data["error"],
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1_score": 0.0,
                    "accuracy": 0.0
                }
            
            # Ensure both are lists
            if not isinstance(pred_data, list):
                pred_data = []
            if not isinstance(truth_data, list):
                truth_data = []
            
            # Extract resource addresses for comparison
            pred_addresses = set(
                resource.get("address", f"{resource.get('type', '')}.{resource.get('name', '')}")
                for resource in pred_data
                if isinstance(resource, dict)
            )
            
            truth_addresses = set(
                resource.get("address", f"{resource.get('type', '')}.{resource.get('name', '')}")
                for resource in truth_data
                if isinstance(resource, dict)
            )
            
            # Calculate metrics
            correct_predictions = len(pred_addresses & truth_addresses)
            total_predictions = len(pred_addresses)
            total_ground_truth = len(truth_addresses)
            
            precision = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            recall = correct_predictions / total_ground_truth if total_ground_truth > 0 else 0.0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            # Calculate detailed comparison
            missing_resources = list(truth_addresses - pred_addresses)
            extra_resources = list(pred_addresses - truth_addresses)
            
            # Structural similarity
            pred_json_str = normalize_json_output(predicted)
            truth_json_str = normalize_json_output(ground_truth)
            text_similarity = calculate_similarity(pred_json_str, truth_json_str)
            
            return {
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "accuracy": correct_predictions / max(total_ground_truth, 1),
                "correct_predictions": correct_predictions,
                "total_predictions": total_predictions,
                "total_ground_truth": total_ground_truth,
                "missing_resources": missing_resources,
                "extra_resources": extra_resources,
                "text_similarity": text_similarity,
                "prediction_valid_json": True,
                "ground_truth_valid_json": True
            }
            
        except json.JSONDecodeError as e:
            return {
                "comparison_error": f"JSON parsing failed: {str(e)}",
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0,
                "prediction_valid_json": False,
                "ground_truth_valid_json": False
            }
        except Exception as e:
            return {
                "comparison_error": str(e),
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "accuracy": 0.0
            }
    
    def validate_on_test_set(self, test_file: str, max_samples: int = 50) -> Dict[str, Any]:
        """Validate model on test dataset"""
        logger.info(f"Validating model on test set: {test_file}")
        
        if self.model is None:
            self.load_trained_model()
        
        # Load test samples
        test_samples = []
        try:
            with jsonlines.open(test_file) as reader:
                for i, sample in enumerate(reader):
                    if i >= max_samples:
                        break
                    test_samples.append(sample)
        except Exception as e:
            logger.error(f"Failed to load test samples: {e}")
            raise
        
        logger.info(f"Validating on {len(test_samples)} test samples")
        
        results = []
        metrics_accumulator = {
            "precision": [],
            "recall": [],
            "f1_score": [],
            "accuracy": [],
            "text_similarity": []
        }
        
        for i, sample in enumerate(test_samples):
            logger.info(f"Validating sample {i+1}/{len(test_samples)}")
            
            try:
                # Get model prediction
                predicted = self.predict_terraform_output(sample["input"])
                
                # Get ground truth (from sample or generate fresh)
                ground_truth = sample["output"]
                
                # Compare results
                comparison = self.compare_predictions(predicted, ground_truth)
                
                # Store result
                result = {
                    "sample_id": sample.get("id", f"test_sample_{i}"),
                    "repository": sample.get("repository", "unknown"),
                    "provider": sample.get("primary_provider", "unknown"),
                    "prediction": predicted,
                    "ground_truth": ground_truth,
                    "metrics": comparison,
                    "metadata": sample.get("metadata", {})
                }
                
                results.append(result)
                
                # Accumulate metrics
                for metric in ["precision", "recall", "f1_score", "accuracy", "text_similarity"]:
                    if metric in comparison:
                        metrics_accumulator[metric].append(comparison[metric])
                
                # Log progress
                if comparison.get("f1_score", 0) > 0:
                    logger.info(f"  Sample {i+1}: F1={comparison['f1_score']:.3f}, "
                               f"Precision={comparison['precision']:.3f}, "
                               f"Recall={comparison['recall']:.3f}")
                else:
                    logger.warning(f"  Sample {i+1}: Prediction failed")
                    
            except Exception as e:
                logger.error(f"Validation failed for sample {i+1}: {e}")
                
                # Add error result
                results.append({
                    "sample_id": sample.get("id", f"test_sample_{i}"),
                    "repository": sample.get("repository", "unknown"),
                    "error": str(e),
                    "metrics": {"precision": 0, "recall": 0, "f1_score": 0, "accuracy": 0}
                })
        
        # Calculate aggregate metrics
        aggregate_metrics = {}
        for metric, values in metrics_accumulator.items():
            if values:
                aggregate_metrics[f"avg_{metric}"] = np.mean(values)
                aggregate_metrics[f"std_{metric}"] = np.std(values)
                aggregate_metrics[f"min_{metric}"] = np.min(values)
                aggregate_metrics[f"max_{metric}"] = np.max(values)
            else:
                aggregate_metrics[f"avg_{metric}"] = 0.0
        
        # Create validation report
        validation_report = {
            "validation_info": {
                "model_path": self.model_path,
                "test_file": test_file,
                "total_samples": len(test_samples),
                "successful_predictions": len([r for r in results if "error" not in r]),
                "validation_date": datetime.now().isoformat()
            },
            "aggregate_metrics": aggregate_metrics,
            "detailed_results": results,
            "provider_performance": self._analyze_provider_performance(results),
            "complexity_analysis": self._analyze_complexity_performance(results)
        }
        
        # Save validation results
        results_file = self.output_dir / f"validation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        safe_file_operation("write", str(results_file), validation_report)
        
        # Create summary
        self._create_validation_summary(validation_report, results_file)
        
        logger.info(f"Validation completed. Results saved to {results_file}")
        logger.info(f"Average F1 Score: {aggregate_metrics['avg_f1_score']:.3f}")
        logger.info(f"Average Precision: {aggregate_metrics['avg_precision']:.3f}")
        logger.info(f"Average Recall: {aggregate_metrics['avg_recall']:.3f}")
        
        return validation_report
    
    def _analyze_provider_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by cloud provider"""
        provider_metrics = {}
        
        for result in results:
            provider = result.get("provider", "unknown")
            metrics = result.get("metrics", {})
            
            if provider not in provider_metrics:
                provider_metrics[provider] = {
                    "samples": 0,
                    "f1_scores": [],
                    "precision_scores": [],
                    "recall_scores": []
                }
            
            provider_metrics[provider]["samples"] += 1
            
            if "f1_score" in metrics:
                provider_metrics[provider]["f1_scores"].append(metrics["f1_score"])
                provider_metrics[provider]["precision_scores"].append(metrics["precision"])
                provider_metrics[provider]["recall_scores"].append(metrics["recall"])
        
        # Calculate averages
        provider_summary = {}
        for provider, data in provider_metrics.items():
            if data["f1_scores"]:
                provider_summary[provider] = {
                    "sample_count": data["samples"],
                    "avg_f1": np.mean(data["f1_scores"]),
                    "avg_precision": np.mean(data["precision_scores"]),
                    "avg_recall": np.mean(data["recall_scores"]),
                    "std_f1": np.std(data["f1_scores"])
                }
            else:
                provider_summary[provider] = {
                    "sample_count": data["samples"],
                    "avg_f1": 0.0,
                    "avg_precision": 0.0,
                    "avg_recall": 0.0,
                    "std_f1": 0.0
                }
        
        return provider_summary
    
    def _analyze_complexity_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by configuration complexity"""
        complexity_buckets = {"low": [], "medium": [], "high": []}
        
        for result in results:
            complexity = result.get("metadata", {}).get("complexity_score", 0)
            f1_score = result.get("metrics", {}).get("f1_score", 0)
            
            if complexity < 15:
                bucket = "low"
            elif complexity < 40:
                bucket = "medium"
            else:
                bucket = "high"
            
            complexity_buckets[bucket].append(f1_score)
        
        complexity_analysis = {}
        for bucket, scores in complexity_buckets.items():
            if scores:
                complexity_analysis[bucket] = {
                    "sample_count": len(scores),
                    "avg_f1": np.mean(scores),
                    "std_f1": np.std(scores)
                }
            else:
                complexity_analysis[bucket] = {
                    "sample_count": 0,
                    "avg_f1": 0.0,
                    "std_f1": 0.0
                }
        
        return complexity_analysis
    
    def _create_validation_summary(self, validation_report: Dict, results_file: Path):
        """Create human-readable validation summary"""
        metrics = validation_report["aggregate_metrics"]
        
        summary = f"""
# Terraform Model Validation Summary

**Validation Date**: {validation_report['validation_info']['validation_date']}
**Model**: {validation_report['validation_info']['model_path']}

## Overall Performance
- **Samples Tested**: {validation_report['validation_info']['total_samples']}
- **Successful Predictions**: {validation_report['validation_info']['successful_predictions']}
- **Average F1 Score**: {metrics['avg_f1_score']:.3f} ± {metrics.get('std_f1_score', 0):.3f}
- **Average Precision**: {metrics['avg_precision']:.3f} ± {metrics.get('std_precision', 0):.3f}
- **Average Recall**: {metrics['avg_recall']:.3f} ± {metrics.get('std_recall', 0):.3f}

## Performance by Provider
"""
        
        for provider, provider_metrics in validation_report["provider_performance"].items():
            summary += f"- **{provider.upper()}**: F1={provider_metrics['avg_f1']:.3f} ({provider_metrics['sample_count']} samples)\n"
        
        summary += "\n## Performance by Complexity\n"
        for complexity, complexity_metrics in validation_report["complexity_analysis"].items():
            summary += f"- **{complexity.title()}**: F1={complexity_metrics['avg_f1']:.3f} ({complexity_metrics['sample_count']} samples)\n"
        
        summary_file = results_file.with_suffix('.md')
        with open(summary_file, 'w') as f:
            f.write(summary)
        
        logger.info(f"Validation summary saved to {summary_file}")
    
    def validate_single_configuration(self, terraform_config: str) -> Dict[str, Any]:
        """Validate model on a single Terraform configuration"""
        logger.info("Validating single configuration")
        
        if self.model is None:
            self.load_trained_model()
        
        try:
            # Get model prediction
            predicted = self.predict_terraform_output(terraform_config)
            
            # Get ground truth
            ground_truth = self.generate_ground_truth(terraform_config)
            
            # Compare results
            comparison = self.compare_predictions(predicted, ground_truth)
            
            result = {
                "configuration": terraform_config,
                "prediction": predicted,
                "ground_truth": ground_truth,
                "metrics": comparison,
                "validation_timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Single validation failed: {e}")
            return {"error": str(e)}
    
    def benchmark_inference_speed(self, test_samples: int = 10) -> Dict[str, float]:
        """Benchmark model inference speed"""
        logger.info("Benchmarking inference speed")
        
        if self.model is None:
            self.load_trained_model()
        
        # Create test configuration
        test_config = """# File: main.tf
resource "aws_instance" "test" {
  ami           = "ami-0c02fb55956c7d316"
  instance_type = "t3.micro"
  
  tags = {
    Name = "test-instance"
  }
}"""
        
        inference_times = []
        
        for i in range(test_samples):
            start_time = time.time()
            
            try:
                prediction = self.predict_terraform_output(test_config, max_new_tokens=1024)
                end_time = time.time()
                
                inference_time = end_time - start_time
                inference_times.append(inference_time)
                
                logger.debug(f"Inference {i+1}: {inference_time:.2f}s")
                
            except Exception as e:
                logger.warning(f"Inference {i+1} failed: {e}")
        
        if inference_times:
            return {
                "avg_inference_time": np.mean(inference_times),
                "std_inference_time": np.std(inference_times),
                "min_inference_time": np.min(inference_times),
                "max_inference_time": np.max(inference_times),
                "total_samples": len(inference_times)
            }
        else:
            return {"error": "All inference attempts failed"}

def main():
    """Main validation execution"""
    import time
    
    validator = TerraformModelValidator()
    
    # Test file path
    test_file = str(config.data_dir / "processed" / "test.jsonl")
    
    if not os.path.exists(test_file):
        logger.error(f"Test file not found: {test_file}")
        print("Please run dataset processing first")
        sys.exit(1)
    
    try:
        # Load model
        validator.load_trained_model()
        
        # Run validation
        results = validator.validate_on_test_set(test_file, max_samples=20)
        
        print("Model validation completed!")
        print(f"Average F1 Score: {results['aggregate_metrics']['avg_f1_score']:.3f}")
        print(f"Results saved to: {validator.output_dir}")
        
        # Run inference benchmark
        speed_results = validator.benchmark_inference_speed()
        print(f"Average inference time: {speed_results.get('avg_inference_time', 0):.2f}s")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
