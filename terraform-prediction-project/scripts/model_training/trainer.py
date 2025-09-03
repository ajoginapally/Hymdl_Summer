"""
Fine-tuning trainer for Terraform prediction model
"""

import os
import json
import torch
import wandb
import jsonlines
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from datasets import Dataset
from transformers import (
    Trainer,
    EarlyStoppingCallback,
    DataCollatorForLanguageModeling
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import gc

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import monitor_system_resources, safe_file_operation
from .model_setup import TerraformModelSetup, TerraformDataCollator

logger = logging.getLogger(__name__)

class TerraformTrainer:
    """Fine-tuning trainer for Terraform prediction"""
    
    def __init__(self):
        self.model_setup = TerraformModelSetup()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup wandb if configured
        self._setup_wandb()
    
    def _setup_wandb(self):
        """Initialize Weights & Biases tracking"""
        if config.wandb_api_key:
            try:
                wandb.init(
                    project=config.wandb_project,
                    entity=config.wandb_entity if config.wandb_entity else None,
                    name=f"terraform-predictor-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    config={
                        "model_name": config.base_model_name,
                        "learning_rate": config.learning_rate,
                        "batch_size": config.batch_size,
                        "gradient_accumulation_steps": config.gradient_accumulation_steps,
                        "num_epochs": config.num_epochs,
                        "max_seq_length": config.max_seq_length,
                        "lora_r": 16,
                        "lora_alpha": 32
                    },
                    tags=["terraform", "infrastructure", "llama", "fine-tuning"]
                )
                logger.info("Weights & Biases initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize wandb: {e}")
    
    def load_dataset(self, dataset_file: str) -> Dataset:
        """Load and prepare dataset for training"""
        logger.info(f"Loading dataset from {dataset_file}")
        
        samples = []
        try:
            with jsonlines.open(dataset_file) as reader:
                for sample in reader:
                    samples.append(sample)
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        logger.info(f"Loaded {len(samples)} samples")
        
        # Convert to HuggingFace Dataset
        dataset_dict = {
            "input": [sample["input"] for sample in samples],
            "output": [sample["output"] for sample in samples],
            "id": [sample.get("id", f"sample_{i}") for i, sample in enumerate(samples)],
            "provider": [sample.get("primary_provider", "unknown") for sample in samples]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        
        # Log dataset statistics
        self._log_dataset_statistics(samples)
        
        return dataset
    
    def _log_dataset_statistics(self, samples: List[Dict]):
        """Log dataset statistics"""
        total_samples = len(samples)
        
        # Provider distribution
        provider_counts = {}
        token_lengths = []
        complexity_scores = []
        
        for sample in samples:
            provider = sample.get("primary_provider", "unknown")
            provider_counts[provider] = provider_counts.get(provider, 0) + 1
            
            metadata = sample.get("metadata", {})
            token_lengths.append(metadata.get("total_token_length", 0))
            complexity_scores.append(metadata.get("complexity_score", 0))
        
        logger.info(f"Dataset statistics:")
        logger.info(f"  Total samples: {total_samples}")
        logger.info(f"  Provider distribution: {provider_counts}")
        logger.info(f"  Avg token length: {np.mean(token_lengths):.1f}")
        logger.info(f"  Max token length: {np.max(token_lengths)}")
        logger.info(f"  Avg complexity: {np.mean(complexity_scores):.1f}")
        
        # Log to wandb if available
        if wandb.run:
            wandb.log({
                "dataset/total_samples": total_samples,
                "dataset/avg_token_length": np.mean(token_lengths),
                "dataset/max_token_length": np.max(token_lengths),
                "dataset/avg_complexity": np.mean(complexity_scores)
            })
    
    def prepare_training_data(self, train_file: str, val_file: str) -> Tuple[Dataset, Dataset]:
        """Prepare training and validation datasets"""
        logger.info("Preparing training datasets")
        
        train_dataset = self.load_dataset(train_file)
        val_dataset = self.load_dataset(val_file)
        
        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        return train_dataset, val_dataset
    
    def setup_trainer(self, train_dataset: Dataset, val_dataset: Dataset) -> Trainer:
        """Setup the Trainer with all components"""
        logger.info("Setting up trainer")
        
        # Load model and tokenizer
        self.model, self.tokenizer = self.model_setup.load_model_and_tokenizer()
        
        # Validate setup
        if not self.model_setup.validate_setup():
            raise RuntimeError("Model setup validation failed")
        
        # Setup training arguments
        training_args = self.model_setup.setup_training_arguments()
        
        # Setup data collator
        data_collator = TerraformDataCollator(\n            tokenizer=self.tokenizer,\n            max_length=config.max_seq_length\n        )\n        \n        # Setup callbacks\n        callbacks = []\n        if training_args.early_stopping_patience > 0:\n            callbacks.append(EarlyStoppingCallback(\n                early_stopping_patience=training_args.early_stopping_patience,\n                early_stopping_threshold=0.001\n            ))\n        \n        # Create trainer\n        self.trainer = Trainer(\n            model=self.model,\n            args=training_args,\n            train_dataset=train_dataset,\n            eval_dataset=val_dataset,\n            tokenizer=self.tokenizer,\n            data_collator=data_collator,\n            callbacks=callbacks\n        )\n        \n        logger.info("Trainer setup completed")\n        return self.trainer\n    \n    def train_model(self, train_file: str, val_file: str) -> Dict[str, Any]:\n        """Main training method"""
        logger.info("Starting model training")\n        \n        try:\n            # Monitor initial resources\n            monitor_system_resources()\n            \n            # Prepare datasets\n            train_dataset, val_dataset = self.prepare_training_data(train_file, val_file)\n            \n            # Setup trainer\n            trainer = self.setup_trainer(train_dataset, val_dataset)\n            \n            # Log training info\n            total_steps = len(train_dataset) * config.num_epochs // (config.batch_size * config.gradient_accumulation_steps)\n            logger.info(f"Training will run for {total_steps} steps over {config.num_epochs} epochs")\n            \n            # Start training\n            logger.info("Beginning fine-tuning...")\n            training_result = trainer.train()\n            \n            # Save final model\n            logger.info("Saving trained model")\n            trainer.save_model()\n            self.tokenizer.save_pretrained(trainer.args.output_dir)\n            \n            # Save training metrics\n            training_metrics = {\n                "training_loss": training_result.training_loss,\n                "training_steps": training_result.global_step,\n                "training_epochs": training_result.epoch,\n                "best_model_checkpoint": getattr(training_result, 'best_model_checkpoint', None),\n                "training_duration_seconds": training_result.metrics.get("train_runtime", 0),\n                "samples_per_second": training_result.metrics.get("train_samples_per_second", 0)\n            }\n            \n            metrics_file = Path(trainer.args.output_dir) / "training_metrics.json"\n            safe_file_operation("write", str(metrics_file), training_metrics)\n            \n            # Final evaluation\n            logger.info("Running final evaluation")\n            eval_results = trainer.evaluate()\n            \n            final_results = {\n                "training_metrics": training_metrics,\n                "eval_metrics": eval_results,\n                "model_path": trainer.args.output_dir\n            }\n            \n            # Save complete results\n            results_file = Path(trainer.args.output_dir) / "training_results.json"\n            safe_file_operation("write", str(results_file), final_results)\n            \n            # Log to wandb\n            if wandb.run:\n                wandb.log({\n                    "training/final_loss": training_result.training_loss,\n                    "training/total_steps": training_result.global_step,\n                    "eval/final_loss": eval_results.get("eval_loss", 0)\n                })\n                wandb.finish()\n            \n            logger.info(f"Training completed successfully!")
            logger.info(f"Model saved to: {trainer.args.output_dir}")
            logger.info(f"Training loss: {training_result.training_loss:.4f}")
            logger.info(f"Validation loss: {eval_results.get('eval_loss', 0):.4f}")
            \n            return final_results\n            \n        except Exception as e:\n            logger.error(f\"Training failed: {e}\")\n            \n            # Cleanup on failure\n            self.model_setup.cleanup_memory()\n            \n            # Close wandb run\n            if wandb.run:\n                wandb.finish()\n            \n            raise\n    \n    def resume_training(self, checkpoint_dir: str, train_file: str, val_file: str) -> Dict[str, Any]:\n        \"\"\"Resume training from checkpoint\"\"\"\n        logger.info(f\"Resuming training from checkpoint: {checkpoint_dir}\")\n        \n        if not os.path.exists(checkpoint_dir):\n            raise ValueError(f\"Checkpoint directory not found: {checkpoint_dir}\")\n        \n        try:\n            # Prepare datasets\n            train_dataset, val_dataset = self.prepare_training_data(train_file, val_file)\n            \n            # Setup trainer\n            trainer = self.setup_trainer(train_dataset, val_dataset)\n            \n            # Resume training\n            training_result = trainer.train(resume_from_checkpoint=checkpoint_dir)\n            \n            # Save results similar to train_model\n            results = {\n                \"resumed_from\": checkpoint_dir,\n                \"training_loss\": training_result.training_loss,\n                \"global_step\": training_result.global_step\n            }\n            \n            return results\n            \n        except Exception as e:\n            logger.error(f\"Failed to resume training: {e}\")\n            raise\n    \n    def evaluate_model(self, model_path: str, test_file: str) -> Dict[str, Any]:\n        \"\"\"Evaluate trained model on test set\"\"\"\n        logger.info(f\"Evaluating model from {model_path}\")\n        \n        try:\n            # Load test dataset\n            test_dataset = self.load_dataset(test_file)\n            \n            # Load model for evaluation\n            from peft import PeftModel\n            \n            base_model = AutoModelForCausalLM.from_pretrained(\n                config.base_model_name,\n                torch_dtype=torch.bfloat16,\n                device_map="auto"\n            )\n            \n            model = PeftModel.from_pretrained(base_model, model_path)\n            tokenizer = AutoTokenizer.from_pretrained(model_path)\n            \n            model.eval()\n            \n            # Run evaluation\n            data_collator = TerraformDataCollator(tokenizer)\n            \n            trainer = Trainer(\n                model=model,\n                tokenizer=tokenizer,\n                data_collator=data_collator\n            )\n            \n            eval_results = trainer.evaluate(eval_dataset=test_dataset)\n            \n            logger.info(f\"Evaluation completed - Loss: {eval_results.get('eval_loss', 0):.4f}\")\n            \n            return eval_results\n            \n        except Exception as e:\n            logger.error(f\"Evaluation failed: {e}\")\n            raise

class TrainingMonitor:\n    \"\"\"Monitor training progress and system resources\"\"\"\n    \n    def __init__(self):\n        self.training_history = []\n        self.resource_history = []\n    \n    def log_training_step(self, step: int, loss: float, learning_rate: float):\n        \"\"\"Log training step information\"\"\"\n        entry = {\n            "step": step,\n            "loss": loss,\n            "learning_rate": learning_rate,\n            "timestamp": datetime.now().isoformat()\n        }\n        \n        self.training_history.append(entry)\n        \n        # Log to wandb if available\n        if wandb.run:\n            wandb.log({\n                "train/loss": loss,\n                "train/learning_rate": learning_rate,\n                "train/step": step\n            })\n    \n    def log_system_resources(self):\n        \"\"\"Log system resource usage\"\"\"\n        try:\n            if torch.cuda.is_available():\n                memory_allocated = torch.cuda.memory_allocated() / (1024**3)\n                memory_reserved = torch.cuda.memory_reserved() / (1024**3)\n                \n                resource_entry = {\n                    "gpu_memory_allocated_gb": memory_allocated,\n                    "gpu_memory_reserved_gb": memory_reserved,\n                    "timestamp": datetime.now().isoformat()\n                }\n                \n                self.resource_history.append(resource_entry)\n                \n                # Log to wandb\n                if wandb.run:\n                    wandb.log({\n                        "system/gpu_memory_allocated": memory_allocated,\n                        "system/gpu_memory_reserved": memory_reserved\n                    })\n                    \n        except Exception as e:\n            logger.warning(f\"Resource monitoring failed: {e}\")\n    \n    def save_training_history(self, output_dir: str):\n        \"\"\"Save complete training history\"\"\"\n        history_data = {\n            "training_history": self.training_history,\n            "resource_history": self.resource_history,\n            "total_steps": len(self.training_history)\n        }\n        \n        history_file = Path(output_dir) / "training_history.json"\n        safe_file_operation("write", str(history_file), history_data)\n        \n        logger.info(f"Training history saved to {history_file}")

class CustomTrainerCallback:\n    \"\"\"Custom callback for enhanced training monitoring\"\"\"\n    \n    def __init__(self, monitor: TrainingMonitor):\n        self.monitor = monitor\n        self.step_count = 0\n    \n    def on_log(self, args, state, control, logs=None, **kwargs):\n        \"\"\"Called when logging\"\"\"\n        if logs:\n            self.step_count += 1\n            \n            # Log training metrics\n            if "loss" in logs:\n                self.monitor.log_training_step(\n                    step=state.global_step,\n                    loss=logs["loss"],\n                    learning_rate=logs.get("learning_rate", 0)\n                )\n            \n            # Log system resources every 10 steps\n            if self.step_count % 10 == 0:\n                self.monitor.log_system_resources()\n    \n    def on_save(self, args, state, control, **kwargs):\n        \"\"\"Called when saving checkpoint\"\"\"\n        logger.info(f\"Checkpoint saved at step {state.global_step}\")\n        \n        # Force memory cleanup\n        if torch.cuda.is_available():\n            torch.cuda.empty_cache()
        gc.collect()

def main():
    \"\"\"Main training execution\"\"\"\n    trainer_instance = TerraformTrainer()\n    \n    # Dataset files\n    train_file = str(config.data_dir / \"processed\" / \"train.jsonl\")\n    val_file = str(config.data_dir / \"processed\" / \"validation.jsonl\")\n    \n    # Check if files exist\n    if not os.path.exists(train_file):\n        logger.error(f\"Training file not found: {train_file}\")\n        print(\"Please run dataset processing first\")\n        sys.exit(1)\n    \n    if not os.path.exists(val_file):\n        logger.error(f\"Validation file not found: {val_file}\")\n        print(\"Please run dataset processing first\")\n        sys.exit(1)\n    \n    try:\n        # Start training\n        results = trainer_instance.train_model(train_file, val_file)\n        \n        print(\"Fine-tuning completed successfully!\")\n        print(f\"Model saved to: {results['model_path']}\")\n        print(f\"Training loss: {results['training_metrics']['training_loss']:.4f}\")\n        print(f\"Validation loss: {results['eval_metrics'].get('eval_loss', 0):.4f}\")\n        \n    except Exception as e:\n        logger.error(f\"Training failed: {e}\")\n        sys.exit(1)

if __name__ == "__main__":\n    main()
