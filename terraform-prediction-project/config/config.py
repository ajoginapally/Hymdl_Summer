"""
Configuration management for Terraform prediction project
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class Config:
    """Central configuration management"""
    
    def __init__(self, env_file: Optional[str] = None):
        # Load environment variables
        if env_file:
            load_dotenv(env_file)
        else:
            # Try to load from .env file in project root
            project_root = Path(__file__).parent.parent
            env_path = project_root / ".env"
            if env_path.exists():
                load_dotenv(env_path)
        
        self.project_root = Path(__file__).parent.parent
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "terraform_prediction.log"),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger(__name__)
    
    # GitHub Configuration
    @property
    def github_token(self) -> str:
        return os.getenv("GITHUB_TOKEN", "")
    
    @property
    def github_api_base_url(self) -> str:
        return os.getenv("GITHUB_API_BASE_URL", "https://api.github.com")
    
    # Weights & Biases Configuration
    @property
    def wandb_api_key(self) -> str:
        return os.getenv("WANDB_API_KEY", "")
    
    @property
    def wandb_project(self) -> str:
        return os.getenv("WANDB_PROJECT", "terraform-prediction")
    
    @property
    def wandb_entity(self) -> str:
        return os.getenv("WANDB_ENTITY", "")
    
    # Hugging Face Configuration
    @property
    def hf_token(self) -> str:
        return os.getenv("HF_TOKEN", "")
    
    @property
    def hf_home(self) -> str:
        return os.getenv("HF_HOME", str(self.project_root / "huggingface_cache"))
    
    # Azure Configuration
    @property
    def azure_client_id(self) -> str:
        return os.getenv("AZURE_CLIENT_ID", "")
    
    @property
    def azure_client_secret(self) -> str:
        return os.getenv("AZURE_CLIENT_SECRET", "")
    
    @property
    def azure_tenant_id(self) -> str:
        return os.getenv("AZURE_TENANT_ID", "")
    
    @property
    def azure_subscription_id(self) -> str:
        return os.getenv("AZURE_SUBSCRIPTION_ID", "")
    
    # Model Configuration
    @property
    def model_name(self) -> str:
        return os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
    
    @property
    def base_model_name(self) -> str:
        return os.getenv("BASE_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
    
    @property
    def model_output_dir(self) -> Path:
        return Path(os.getenv("MODEL_OUTPUT_DIR", str(self.project_root / "models" / "terraform-predictor")))
    
    # Training Configuration
    @property
    def max_repos(self) -> int:
        return int(os.getenv("MAX_REPOS", "100"))
    
    @property
    def max_samples_per_repo(self) -> int:
        return int(os.getenv("MAX_SAMPLES_PER_REPO", "10"))
    
    @property
    def batch_size(self) -> int:
        return int(os.getenv("BATCH_SIZE", "1"))
    
    @property
    def gradient_accumulation_steps(self) -> int:
        return int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "8"))
    
    @property
    def learning_rate(self) -> float:
        return float(os.getenv("LEARNING_RATE", "2e-4"))
    
    @property
    def num_epochs(self) -> int:
        return int(os.getenv("NUM_EPOCHS", "3"))
    
    @property
    def max_seq_length(self) -> int:
        return int(os.getenv("MAX_SEQ_LENGTH", "32768"))
    
    # Resource Limits
    @property
    def max_workers(self) -> int:
        return int(os.getenv("MAX_WORKERS", "4"))
    
    @property
    def memory_limit_gb(self) -> int:
        return int(os.getenv("MEMORY_LIMIT_GB", "32"))
    
    @property
    def gpu_memory_fraction(self) -> float:
        return float(os.getenv("GPU_MEMORY_FRACTION", "0.9"))
    
    # Directory Paths
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def scripts_dir(self) -> Path:
        return self.project_root / "scripts"
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    # Server Configuration
    @property
    def remote_server(self) -> str:
        return os.getenv("REMOTE_SERVER", "manthram.tplinkdns.com")
    
    @property
    def remote_port(self) -> int:
        return int(os.getenv("REMOTE_PORT", "999"))
    
    @property
    def remote_user(self) -> str:
        return os.getenv("REMOTE_USER", "arnav")
    
    @property
    def remote_project_path(self) -> str:
        return os.getenv("REMOTE_PROJECT_PATH", "/home/arnav/terraform-prediction-project")
    
    # Target Services
    @property
    def aws_services(self) -> list:
        return [
            "ec2", "s3", "rds", "lambda", "iam", "vpc", "elb", "autoscaling",
            "cloudwatch", "sns", "sqs", "dynamodb", "cloudfront", "route53",
            "apigateway", "eks", "ecs", "ecr", "elasticache", "redshift",
            "kinesis", "glue", "athena", "emr", "sagemaker", "cognito",
            "secretsmanager", "ssm", "cloudformation", "kms", "acm", "wafv2",
            "shield", "guardduty", "inspector", "config", "cloudtrail",
            "organizations", "backup", "datasync", "transfer", "workspaces",
            "appstream", "workmail", "connect", "pinpoint", "ses", "workdocs"
        ]
    
    @property
    def azure_services(self) -> list:
        return [
            "resource_group", "virtual_network", "subnet", "network_security_group",
            "public_ip", "network_interface", "virtual_machine", "storage_account",
            "storage_blob", "key_vault", "app_service", "app_service_plan",
            "sql_server", "sql_database", "cosmosdb_account", "redis_cache",
            "service_bus_namespace", "service_bus_queue", "application_insights",
            "log_analytics_workspace", "container_registry", "kubernetes_cluster",
            "function_app", "logic_app", "data_factory", "synapse_workspace",
            "machine_learning_workspace", "cognitive_account", "search_service",
            "event_hub_namespace", "event_hub", "stream_analytics_job"
        ]
    
    def validate_config(self) -> bool:
        """Validate required configuration"""
        required_configs = [
            ("GITHUB_TOKEN", self.github_token),
            ("MODEL_NAME", self.model_name),
        ]
        
        missing = []
        for name, value in required_configs:
            if not value:
                missing.append(name)
        
        if missing:
            self.logger.error(f"Missing required configuration: {', '.join(missing)}")
            return False
        
        return True
    
    def get_runtime_info(self) -> Dict[str, Any]:
        """Get runtime information"""
        import torch
        import psutil
        
        return {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_count": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": os.sys.version,
            "project_root": str(self.project_root)
        }

# Global configuration instance
config = Config()
