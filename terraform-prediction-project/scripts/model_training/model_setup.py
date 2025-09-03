"""
Model setup and configuration for Llama-3.2-3B-Instruct fine-tuning
"""

import os
import torch
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
import gc

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import check_gpu_resources, monitor_system_resources, setup_environment_variables

logger = logging.getLogger(__name__)

class TerraformModelSetup:
    """Setup and configure Llama-3.2-3B-Instruct for Terraform prediction"""
    
    def __init__(self):
        self.model_name = config.base_model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.tokenizer = None
        
        # Setup optimal environment
        setup_environment_variables()
        
        # Log system information
        self._log_system_info()
    
    def _log_system_info(self):
        """Log system and GPU information"""
        logger.info(f"Device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"CUDA Version: {torch.version.cuda}")
            logger.info(f"PyTorch Version: {torch.__version__}")
            
            gpu_info = check_gpu_resources()
            if gpu_info.get("available"):
                for gpu_id, info in gpu_info["gpus"].items():
                    logger.info(f"{gpu_id}: {info['name']} ({info['total_memory_gb']:.1f}GB)")
        else:
            logger.warning("CUDA not available - training will be very slow on CPU")
    
    def setup_quantization_config(self) -> BitsAndBytesConfig:
        """Configure 4-bit quantization for efficient training"""
        logger.info("Setting up 4-bit quantization configuration")
        
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage_dtype=torch.bfloat16
        )
    
    def setup_lora_config(self) -> LoraConfig:
        """Configure LoRA for parameter-efficient fine-tuning"""
        logger.info("Setting up LoRA configuration")
        
        # Llama-3.2 specific target modules
        target_modules = [
            "q_proj",
            "k_proj", 
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ]
        
        return LoraConfig(
            r=16,  # Rank of adaptation
            lora_alpha=32,  # LoRA scaling parameter
            target_modules=target_modules,
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False
        )
    
    def load_model_and_tokenizer(self) -> Tuple[Any, Any]:
        """Load and configure model and tokenizer"""
        logger.info(f"Loading model: {self.model_name}")
        
        try:
            # Setup quantization config
            quantization_config = self.setup_quantization_config()
            
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                use_fast=True
            )
            
            # Configure tokenizer
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            
            # Set chat template if not present (for instruction following)
            if not hasattr(self.tokenizer, 'chat_template') or self.tokenizer.chat_template is None:
                self.tokenizer.chat_template = \"\"\"{{ bos_token }}{% for message in messages %}{% if message['role'] == 'user' %}{{ 'User: ' + message['content'] + '\\n\\n' }}{% elif message['role'] == 'assistant' %}{{ 'Assistant: ' + message['content'] + eos_token }}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ 'Assistant: ' }}{% endif %}\"\"\"\n            \n            logger.info(f\"Tokenizer loaded - Vocab size: {len(self.tokenizer)}\")\n            \n            # Load model with quantization\n            logger.info("Loading model with quantization...")
            self.model = AutoModelForCausalLM.from_pretrained(\n                self.model_name,\n                quantization_config=quantization_config,\n                device_map="auto",\n                torch_dtype=torch.bfloat16,\n                trust_remote_code=True,\n                attn_implementation="flash_attention_2" if self._supports_flash_attention() else "eager",\n                use_cache=False  # Disable cache for training\n            )\n            \n            # Prepare model for k-bit training\n            logger.info("Preparing model for k-bit training...")\n            self.model = prepare_model_for_kbit_training(self.model)\n            \n            # Apply LoRA\n            logger.info("Applying LoRA configuration...")\n            lora_config = self.setup_lora_config()\n            self.model = get_peft_model(self.model, lora_config)\n            \n            # Enable gradient checkpointing\n            self.model.gradient_checkpointing_enable()\n            \n            # Print trainable parameters\n            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)\n            total_params = sum(p.numel() for p in self.model.parameters())\n            \n            logger.info(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
            logger.info(f"Total parameters: {total_params:,}")
            \n            return self.model, self.tokenizer\n            \n        except Exception as e:\n            logger.error(f"Failed to load model: {e}")\n            raise\n    \n    def _supports_flash_attention(self) -> bool:\n        """Check if Flash Attention 2 is supported"""
        try:\n            import flash_attn\n            return True\n        except ImportError:\n            logger.info("Flash Attention 2 not available, using standard attention")\n            return False\n    \n    def setup_training_arguments(self, output_dir: Optional[str] = None) -> TrainingArguments:\n        """Setup optimized training arguments for Llama-3.2-3B"""
        if output_dir is None:\n            output_dir = str(config.model_output_dir)\n        \n        # Ensure output directory exists\n        Path(output_dir).mkdir(parents=True, exist_ok=True)\n        \n        logger.info("Setting up training arguments")\n        \n        return TrainingArguments(\n            # Output and logging\n            output_dir=output_dir,\n            logging_dir=str(config.logs_dir / "training"),\n            logging_strategy="steps",\n            logging_steps=10,\n            save_strategy="steps",\n            save_steps=500,\n            save_total_limit=3,\n            \n            # Evaluation\n            eval_strategy="steps",\n            eval_steps=100,\n            load_best_model_at_end=True,\n            metric_for_best_model="eval_loss",\n            greater_is_better=False,\n            \n            # Training parameters\n            num_train_epochs=config.num_epochs,\n            per_device_train_batch_size=config.batch_size,\n            per_device_eval_batch_size=config.batch_size,\n            gradient_accumulation_steps=config.gradient_accumulation_steps,\n            \n            # Optimization\n            learning_rate=config.learning_rate,\n            lr_scheduler_type="cosine",\n            warmup_steps=100,\n            weight_decay=0.01,\n            optim="adamw_torch",\n            adam_beta1=0.9,\n            adam_beta2=0.95,\n            adam_epsilon=1e-8,\n            max_grad_norm=1.0,\n            \n            # Precision and performance\n            bf16=True if torch.cuda.is_available() else False,\n            fp16=False,  # Use bf16 instead\n            dataloader_drop_last=True,\n            dataloader_num_workers=2,\n            gradient_checkpointing=True,\n            \n            # Early stopping\n            early_stopping_patience=3,\n            \n            # Monitoring\n            report_to="wandb" if config.wandb_api_key else None,\n            run_name=f"terraform-predictor-{datetime.now().strftime('%Y%m%d_%H%M%S')}",\n            \n            # Misc\n            remove_unused_columns=False,\n            label_smoothing_factor=0.1,\n            disable_tqdm=False,\n            prediction_loss_only=True\n        )\n    \n    def setup_data_collator(self) -> DataCollatorForLanguageModeling:\n        """Setup data collator for causal language modeling"""
        return DataCollatorForLanguageModeling(\n            tokenizer=self.tokenizer,\n            mlm=False,  # Causal LM, not masked LM\n            return_tensors="pt",\n            pad_to_multiple_of=8  # Optimize for tensor cores\n        )\n    \n    def optimize_model_for_training(self) -> None:\n        """Apply additional optimizations for training"""
        if self.model is None:\n            raise ValueError("Model must be loaded first")\n        \n        logger.info("Applying training optimizations")\n        \n        # Enable gradient checkpointing\n        if hasattr(self.model, 'gradient_checkpointing_enable'):\n            self.model.gradient_checkpointing_enable()\n        \n        # Optimize for memory\n        if torch.cuda.is_available():\n            # Clear cache\n            torch.cuda.empty_cache()\n            \n            # Set memory fraction\n            if hasattr(torch.cuda, 'set_memory_fraction'):\n                torch.cuda.set_memory_fraction(config.gpu_memory_fraction)\n        \n        # Enable compilation if available (PyTorch 2.0+)\n        if hasattr(torch, 'compile') and torch.cuda.is_available():\n            try:\n                logger.info("Compiling model for faster training")\n                self.model = torch.compile(self.model, mode="reduce-overhead")\n            except Exception as e:\n                logger.warning(f"Model compilation failed: {e}")\n    \n    def get_model_memory_usage(self) -> Dict[str, float]:\n        """Get current model memory usage"""
        if self.model is None or not torch.cuda.is_available():\n            return {}\n        \n        model_memory = 0\n        for param in self.model.parameters():\n            model_memory += param.numel() * param.element_size()\n        \n        model_memory_gb = model_memory / (1024**3)\n        \n        return {\n            "model_memory_gb": model_memory_gb,\n            "gpu_memory_allocated_gb": torch.cuda.memory_allocated() / (1024**3),\n            "gpu_memory_reserved_gb": torch.cuda.memory_reserved() / (1024**3)\n        }\n    \n    def validate_setup(self) -> bool:\n        """Validate that model setup is correct"""
        logger.info("Validating model setup")\n        \n        try:\n            # Check model and tokenizer are loaded\n            if self.model is None or self.tokenizer is None:\n                logger.error("Model or tokenizer not loaded")\n                return False\n            \n            # Test tokenization\n            test_input = "resource \"aws_instance\" \"test\" { ami = \"ami-123\" }"\n            tokens = self.tokenizer.encode(test_input)\n            \n            if len(tokens) == 0:\n                logger.error("Tokenization failed")\n                return False\n            \n            # Test model forward pass (small input)\n            if torch.cuda.is_available():\n                input_ids = torch.tensor([tokens[:10]]).to(self.device)\n                \n                with torch.no_grad():\n                    outputs = self.model(input_ids)\n                \n                if outputs.logits is None:\n                    logger.error("Model forward pass failed")\n                    return False\n            \n            # Check memory usage\n            memory_info = self.get_model_memory_usage()\n            if memory_info:\n                logger.info(f"Model memory usage: {memory_info['model_memory_gb']:.2f}GB")\n            \n            # Validate LoRA configuration\n            if hasattr(self.model, 'peft_config'):\n                logger.info("LoRA configuration validated")\n            else:\n                logger.warning("LoRA not properly applied")\n            \n            logger.info("Model setup validation successful")\n            return True\n            \n        except Exception as e:\n            logger.error(f"Model setup validation failed: {e}")\n            return False\n    \n    def save_model_config(self, output_dir: str):\n        """Save model configuration for reproducibility"""
        config_dict = {\n            "base_model": self.model_name,\n            "quantization": {\n                "load_in_4bit": True,\n                "bnb_4bit_quant_type": "nf4",\n                "bnb_4bit_use_double_quant": True,\n                "bnb_4bit_compute_dtype": "bfloat16"\n            },\n            "lora": {\n                "r": 16,\n                "lora_alpha": 32,\n                "lora_dropout": 0.1,\n                "bias": "none",\n                "task_type": "CAUSAL_LM"\n            },\n            "training": {\n                "max_seq_length": config.max_seq_length,\n                "learning_rate": config.learning_rate,\n                "batch_size": config.batch_size,\n                "gradient_accumulation_steps": config.gradient_accumulation_steps,\n                "num_epochs": config.num_epochs\n            },\n            "system_info": {\n                "torch_version": torch.__version__,\n                "cuda_available": torch.cuda.is_available(),\n                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0\n            }\n        }\n        \n        config_file = Path(output_dir) / "model_config.json"\n        config_file.parent.mkdir(parents=True, exist_ok=True)\n        \n        with open(config_file, 'w') as f:\n            json.dump(config_dict, f, indent=2)\n        \n        logger.info(f"Model configuration saved to {config_file}")\n    \n    def cleanup_memory(self):\n        """Clean up GPU memory"""
        if torch.cuda.is_available():\n            torch.cuda.empty_cache()\n            torch.cuda.synchronize()\n        \n        # Force garbage collection\n        gc.collect()\n        \n        logger.info("Memory cleanup completed")\n\nclass TerraformDataCollator:\n    """Custom data collator for Terraform prediction training"""
    \n    def __init__(self, tokenizer, max_length: int = None):\n        self.tokenizer = tokenizer\n        self.max_length = max_length or config.max_seq_length\n    \n    def __call__(self, batch) -> Dict[str, torch.Tensor]:\n        """Collate batch for training"""
        # Extract inputs and outputs\n        inputs = [item["input"] for item in batch]\n        outputs = [item["output"] for item in batch]\n        \n        # Combine input and output for causal LM training\n        combined_texts = []\n        for inp, out in zip(inputs, outputs):\n            combined_text = f"{inp}\\n{out}{self.tokenizer.eos_token}"\n            combined_texts.append(combined_text)\n        \n        # Tokenize\n        tokenized = self.tokenizer(\n            combined_texts,\n            padding=True,\n            truncation=True,\n            max_length=self.max_length,\n            return_tensors="pt"\n        )\n        \n        # For causal LM, labels are input_ids shifted\n        labels = tokenized["input_ids"].clone()\n        \n        # Mask padding tokens in labels\n        labels[labels == self.tokenizer.pad_token_id] = -100\n        \n        # Mask input portion (we only want to predict the output)\n        for i, (inp, out) in enumerate(zip(inputs, outputs)):\n            input_length = len(self.tokenizer.encode(inp, add_special_tokens=False))\n            if input_length < labels.shape[1]:\n                labels[i, :input_length] = -100\n        \n        return {\n            "input_ids": tokenized["input_ids"],\n            "attention_mask": tokenized["attention_mask"],\n            "labels": labels\n        }

def main():\n    """Test model setup"""
    setup = TerraformModelSetup()\n    \n    try:\n        # Load model and tokenizer\n        model, tokenizer = setup.load_model_and_tokenizer()\n        \n        # Validate setup\n        if setup.validate_setup():\n            print("Model setup completed successfully!")\n            \n            # Get training arguments\n            training_args = setup.setup_training_arguments()\n            print(f"Training will save to: {training_args.output_dir}")\n            \n            # Save configuration\n            setup.save_model_config(training_args.output_dir)\n            print("Model configuration saved")\n            \n        else:\n            print("Model setup validation failed")\n            \n    except Exception as e:\n        logger.error(f"Model setup failed: {e}")\n        sys.exit(1)\n    \n    finally:\n        setup.cleanup_memory()\n\nif __name__ == "__main__":\n    main()
