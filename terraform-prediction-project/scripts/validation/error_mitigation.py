"""
Error mitigation loop for improving model performance
"""

import json
import jsonlines
from typing import List, Dict, Any
from pathlib import Path
import logging

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import safe_file_operation
from .model_validator import TerraformModelValidator

logger = logging.getLogger(__name__)

class ErrorMitigationLoop:
    """Identify and mitigate model errors through data augmentation"""
    
    def __init__(self):
        self.validator = TerraformModelValidator()
        
    def analyze_error_patterns(self, validation_results: List[Dict]) -> Dict[str, Any]:
        """Analyze validation results to identify error patterns"""
        error_patterns = {
            "provider_errors": {"aws": 0, "azure": 0},
            "missing_constructs": set(),
            "low_performance_types": [],
            "complexity_issues": {"low": 0, "medium": 0, "high": 0}
        }
        
        for result in validation_results:
            metrics = result.get("metrics", {})
            f1_score = metrics.get("f1_score", 0)
            
            if f1_score < 0.5:  # Poor performance threshold
                provider = result.get("provider", "unknown")
                if provider in error_patterns["provider_errors"]:
                    error_patterns["provider_errors"][provider] += 1
                
                # Check for missing constructs
                input_text = result.get("prediction", "")
                if "dynamic" in input_text.lower():
                    error_patterns["missing_constructs"].add("dynamic_blocks")
                if "count" in input_text.lower():
                    error_patterns["missing_constructs"].add("count")
                if "for_each" in input_text.lower():
                    error_patterns["missing_constructs"].add("for_each")
        
        error_patterns["missing_constructs"] = list(error_patterns["missing_constructs"])
        return error_patterns
    
    def generate_augmentation_samples(self, error_patterns: Dict[str, Any]) -> List[Dict]:
        """Generate synthetic samples to address error patterns"""
        augmentation_samples = []
        
        # Add Azure samples if Azure performance is poor
        if error_patterns["provider_errors"]["azure"] > 3:
            augmentation_samples.extend(self._create_azure_samples())
        
        # Add dynamic block samples
        if "dynamic_blocks" in error_patterns["missing_constructs"]:
            augmentation_samples.extend(self._create_dynamic_samples())
        
        # Add count samples
        if "count" in error_patterns["missing_constructs"]:
            augmentation_samples.extend(self._create_count_samples())
        
        return augmentation_samples
    
    def _create_azure_samples(self) -> List[Dict]:
        """Create additional Azure training samples"""
        return [{
            "id": "augment_azure_001",
            "input": """# File: main.tf
resource "azurerm_app_service_plan" "example" {
  name                = "example-appserviceplan"
  location            = "East US"
  resource_group_name = "example-resources"
  
  sku {
    tier = "Standard"
    size = "S1"
  }
}

resource "azurerm_app_service" "example" {
  name                = "example-app-service"
  location            = "East US"
  resource_group_name = "example-resources"
  app_service_plan_id = azurerm_app_service_plan.example.id
}""",
            "output": json.dumps([{
                "address": "azurerm_app_service_plan.example",
                "type": "azurerm_app_service_plan",
                "name": "example",
                "provider_name": "registry.terraform.io/hashicorp/azurerm",
                "change": {
                    "actions": ["create"],
                    "after": {
                        "name": "example-appserviceplan",
                        "location": "East US",
                        "resource_group_name": "example-resources",
                        "sku": [{
                            "tier": "Standard",
                            "size": "S1"
                        }]
                    }
                }
            }]),
            "metadata": {"synthetic": True, "resource_count": 2}
        }]
    
    def _create_dynamic_samples(self) -> List[Dict]:
        """Create dynamic block training samples"""
        return []  # Implementation details...
    
    def _create_count_samples(self) -> List[Dict]:
        """Create count construct training samples"""
        return []  # Implementation details...
    
    def run_mitigation_cycle(self, validation_file: str) -> bool:
        """Run one cycle of error mitigation"""
        with open(validation_file) as f:
            validation_data = json.load(f)
        
        results = validation_data["detailed_results"]
        error_patterns = self.analyze_error_patterns(results)
        
        augmentation_samples = self.generate_augmentation_samples(error_patterns)
        
        if augmentation_samples:
            # Add to training dataset
            train_file = config.data_dir / "processed" / "train.jsonl"
            with jsonlines.open(train_file, 'a') as writer:
                for sample in augmentation_samples:
                    writer.write(sample)
            
            logger.info(f"Added {len(augmentation_samples)} augmentation samples")
            return True
        
        return False

def main():
    loop = ErrorMitigationLoop()
    # Implementation...

if __name__ == "__main__":
    main()
