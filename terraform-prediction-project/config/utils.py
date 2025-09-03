"""
Common utilities and helper functions
"""

import os
import json
import time
import hashlib
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from functools import wraps
import psutil

logger = logging.getLogger(__name__)

def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator

def safe_file_operation(operation: str, file_path: str, content: Any = None, encoding: str = 'utf-8'):
    """Safely perform file operations with error handling"""
    try:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if operation == 'write':
            if isinstance(content, (dict, list)):
                content = json.dumps(content, indent=2, ensure_ascii=False)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            logger.info(f"Successfully wrote to {file_path}")
            
        elif operation == 'read':
            with open(path, 'r', encoding=encoding) as f:
                content = f.read()
            return content
            
        elif operation == 'append':
            with open(path, 'a', encoding=encoding) as f:
                if isinstance(content, (dict, list)):
                    content = json.dumps(content, ensure_ascii=False)
                f.write(content + '\n')
            logger.info(f"Successfully appended to {file_path}")
            
    except Exception as e:
        logger.error(f"File operation '{operation}' failed for {file_path}: {e}")
        raise

def check_system_resources() -> Dict[str, Any]:
    """Check available system resources"""
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "memory": {
            "total_gb": memory.total / (1024**3),
            "available_gb": memory.available / (1024**3),
            "percent_used": memory.percent
        },
        "disk": {
            "total_gb": disk.total / (1024**3),
            "free_gb": disk.free / (1024**3),
            "percent_used": (disk.used / disk.total) * 100
        },
        "cpu_count": psutil.cpu_count(),
        "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
    }

def check_gpu_resources() -> Dict[str, Any]:
    """Check GPU resources if available"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False}
        
        gpu_info = {}
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)
            memory_total = props.total_memory / (1024**3)
            
            gpu_info[f"gpu_{i}"] = {
                "name": props.name,
                "total_memory_gb": memory_total,
                "allocated_gb": memory_allocated,
                "reserved_gb": memory_reserved,
                "free_gb": memory_total - memory_reserved
            }
        
        return {"available": True, "gpus": gpu_info}
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}

def clean_terraform_content(content: str) -> str:
    """Clean and normalize Terraform content"""
    lines = []
    in_multiline_comment = False
    
    for line in content.split('\n'):
        # Handle multi-line comments
        if '/*' in line:
            in_multiline_comment = True
        if '*/' in line:
            in_multiline_comment = False
            continue
        if in_multiline_comment:
            continue
            
        # Remove single-line comments
        if line.strip().startswith('#'):
            continue
        if '//' in line:
            line = line.split('//')[0].rstrip()
        
        # Preserve non-empty lines and normalize whitespace
        line = line.rstrip()
        if line:
            lines.append(line)
    
    return '\n'.join(lines)

def extract_resources_from_terraform(content: str) -> Tuple[List[str], List[str]]:
    """Extract AWS and Azure resources from Terraform content"""
    aws_pattern = r'resource\s+"(aws_\w+)"\s+"(\w+)"'
    azure_pattern = r'resource\s+"(azurerm_\w+)"\s+"(\w+)"'
    
    aws_resources = re.findall(aws_pattern, content, re.IGNORECASE)
    azure_resources = re.findall(azure_pattern, content, re.IGNORECASE)
    
    aws_types = [match[0] for match in aws_resources]
    azure_types = [match[0] for match in azure_resources]
    
    return aws_types, azure_types

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file"""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        logger.error(f"Failed to calculate hash for {file_path}: {e}")
        return ""

def run_terraform_command(command: List[str], cwd: str, timeout: int = 600) -> Tuple[bool, str, str]:
    """Run Terraform command with proper error handling"""
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, 'TF_INPUT': 'false', 'TF_IN_AUTOMATION': 'true'}
        )
        
        success = result.returncode == 0
        return success, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        logger.error(f"Terraform command timed out after {timeout}s: {' '.join(command)}")
        return False, "", "Command timed out"
    except Exception as e:
        logger.error(f"Failed to run terraform command: {e}")
        return False, "", str(e)

def validate_terraform_directory(tf_dir: str) -> bool:
    """Validate that directory contains valid Terraform files"""
    path = Path(tf_dir)
    if not path.exists():
        return False
    
    tf_files = list(path.glob("*.tf"))
    if not tf_files:
        return False
    
    # Check for basic Terraform syntax
    for tf_file in tf_files:
        try:
            content = tf_file.read_text()
            # Basic validation - check for resource or data blocks
            if not (re.search(r'\bresource\s+"', content) or re.search(r'\bdata\s+"', content)):
                continue
            return True
        except Exception:
            continue
    
    return False

def normalize_json_output(json_str: str) -> str:
    """Normalize JSON output for comparison"""
    try:
        data = json.loads(json_str)
        return json.dumps(data, sort_keys=True, separators=(',', ':'))
    except json.JSONDecodeError:
        return json_str

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
    from difflib import SequenceMatcher
    return SequenceMatcher(None, text1, text2).ratio()

def create_backup(file_path: str) -> str:
    """Create backup of file"""
    path = Path(file_path)
    if not path.exists():
        return ""
    
    backup_path = path.with_suffix(f"{path.suffix}.backup.{int(time.time())}")
    try:
        import shutil
        shutil.copy2(path, backup_path)
        logger.info(f"Created backup: {backup_path}")
        return str(backup_path)
    except Exception as e:
        logger.error(f"Failed to create backup for {file_path}: {e}")
        return ""

def cleanup_temp_files(temp_dirs: List[str]):
    """Clean up temporary directories"""
    import shutil
    for temp_dir in temp_dirs:
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up {temp_dir}: {e}")

def format_bytes(bytes_value: int) -> str:
    """Format bytes into human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def estimate_training_time(dataset_size: int, batch_size: int, epochs: int) -> str:
    """Estimate training time based on dataset size"""
    # Rough estimation based on typical training speeds
    steps_per_epoch = dataset_size // batch_size
    total_steps = steps_per_epoch * epochs
    
    # Assume ~2 seconds per step for 3B model on decent GPU
    estimated_seconds = total_steps * 2
    
    hours = estimated_seconds // 3600
    minutes = (estimated_seconds % 3600) // 60
    
    return f"~{hours}h {minutes}m"

def monitor_system_resources():
    """Monitor and log system resource usage"""
    resources = check_system_resources()
    gpu_resources = check_gpu_resources()
    
    logger.info(f"System Resources - Memory: {resources['memory']['percent_used']:.1f}% used, "
               f"Disk: {resources['disk']['percent_used']:.1f}% used")
    
    if gpu_resources.get("available"):
        for gpu_id, gpu_info in gpu_resources["gpus"].items():
            used_pct = (gpu_info["reserved_gb"] / gpu_info["total_memory_gb"]) * 100
            logger.info(f"GPU {gpu_id}: {used_pct:.1f}% memory used ({gpu_info['name']})")

class ProgressTracker:
    """Track progress of long-running operations"""
    
    def __init__(self, total_items: int, description: str = "Processing"):
        self.total_items = total_items
        self.completed_items = 0
        self.description = description
        self.start_time = time.time()
        
    def update(self, increment: int = 1):
        """Update progress"""
        self.completed_items += increment
        percentage = (self.completed_items / self.total_items) * 100
        elapsed_time = time.time() - self.start_time
        
        if self.completed_items > 0:
            eta = (elapsed_time / self.completed_items) * (self.total_items - self.completed_items)
            logger.info(f"{self.description}: {self.completed_items}/{self.total_items} "
                       f"({percentage:.1f}%) - ETA: {eta/60:.1f}m")
    
    def finish(self):
        """Mark as finished"""
        elapsed_time = time.time() - self.start_time
        logger.info(f"{self.description} completed in {elapsed_time/60:.1f} minutes")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for cross-platform compatibility"""
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    # Remove extra underscores and spaces
    filename = re.sub(r'[_\s]+', '_', filename)
    filename = filename.strip('_')
    
    return filename[:255]  # Limit length

def parse_terraform_variables(content: str) -> Dict[str, Any]:
    """Parse Terraform variables from content"""
    variables = {}
    
    # Simple regex-based parsing for common variable patterns
    var_patterns = [
        r'variable\s+"([^"]+)"\s*\{[^}]*default\s*=\s*([^}]+)\}',
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*([^#\n]+)'
    ]
    
    for pattern in var_patterns:
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        for name, value in matches:
            try:
                # Try to parse as JSON/HCL value
                value = value.strip().rstrip(',').rstrip()
                if value.startswith('"') and value.endswith('"'):
                    variables[name] = value[1:-1]  # String value
                elif value.startswith('[') and value.endswith(']'):
                    # List value - simplified parsing
                    variables[name] = value
                elif value.isdigit():
                    variables[name] = int(value)
                elif value.lower() in ['true', 'false']:
                    variables[name] = value.lower() == 'true'
                else:
                    variables[name] = value
            except Exception:
                variables[name] = value
    
    return variables

def generate_sample_id(repo_name: str, directory: str, file_hash: str) -> str:
    """Generate unique sample ID"""
    combined = f"{repo_name}:{directory}:{file_hash}"
    return hashlib.md5(combined.encode()).hexdigest()

def is_terraform_directory(directory: str) -> bool:
    """Check if directory contains Terraform files"""
    path = Path(directory)
    return any(path.glob("*.tf"))

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / (1024 * 1024),  # Resident Set Size
        "vms_mb": memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        "percent": process.memory_percent()
    }

def setup_environment_variables():
    """Setup environment variables for optimal performance"""
    # PyTorch settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    
    # Transformers settings
    os.environ['TRANSFORMERS_CACHE'] = './huggingface_cache'
    os.environ['TRANSFORMERS_OFFLINE'] = '0'
    
    # Tensorflow settings (if used)
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    
    # HF Hub settings
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    
    logger.info("Environment variables configured for optimal performance")

class FileCache:
    """Simple file-based cache for expensive operations"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get cache file path for key"""
        safe_key = sanitize_filename(key)
        return self.cache_dir / f"{safe_key}.json"
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        cache_path = self._get_cache_path(key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache for {key}: {e}")
        return None
    
    def set(self, key: str, value: Any, expiry_hours: int = 24):
        """Set item in cache with expiry"""
        cache_path = self._get_cache_path(key)
        cache_data = {
            "value": value,
            "timestamp": time.time(),
            "expiry_hours": expiry_hours
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache for {key}: {e}")
    
    def is_valid(self, key: str) -> bool:
        """Check if cached item is still valid"""
        cache_path = self._get_cache_path(key)
        if not cache_path.exists():
            return False
        
        try:
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            
            expiry_time = cache_data["timestamp"] + (cache_data["expiry_hours"] * 3600)
            return time.time() < expiry_time
        except Exception:
            return False

# Global cache instance
cache = FileCache()
