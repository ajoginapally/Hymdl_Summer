"""
Advanced Terraform file analyzer for parsing HCL content
"""

import os
import re
import json
import hcl2
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import clean_terraform_content, parse_terraform_variables

logger = logging.getLogger(__name__)

class TerraformAnalyzer:
    """Advanced analyzer for Terraform configurations"""
    
    def __init__(self):
        self.aws_resource_pattern = re.compile(r'resource\s+"(aws_\w+)"\s+"([^"]+)"')
        self.azure_resource_pattern = re.compile(r'resource\s+"(azurerm_\w+)"\s+"([^"]+)"')
        self.variable_pattern = re.compile(r'variable\s+"([^"]+)"\s*\{([^}]*)\}', re.DOTALL)
        self.locals_pattern = re.compile(r'locals\s*\{([^}]*)\}', re.DOTALL)
        self.output_pattern = re.compile(r'output\s+"([^"]+)"\s*\{([^}]*)\}', re.DOTALL)
        self.module_pattern = re.compile(r'module\s+"([^"]+)"\s*\{([^}]*)\}', re.DOTALL)
        self.data_pattern = re.compile(r'data\s+"([^"]+)"\s+"([^"]+)"\s*\{([^}]*)\}', re.DOTALL)
    
    def parse_terraform_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a single Terraform file with comprehensive analysis"""
        result = {
            "file_path": file_path,
            "resources": [],
            "variables": [],
            "locals": [],
            "outputs": [],
            "modules": [],
            "data_sources": [],
            "provider_config": [],
            "analysis": {
                "has_dynamic_blocks": False,
                "has_count": False,
                "has_for_each": False,
                "complexity_score": 0,
                "line_count": 0,
                "resource_count": 0
            }
        }
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            result["analysis"]["line_count"] = len(content.split('\n'))
            
            # Clean content for analysis
            cleaned_content = clean_terraform_content(content)
            
            # Try HCL2 parsing first (more accurate)
            try:
                parsed_hcl = hcl2.loads(content)
                result.update(self._parse_hcl2_structure(parsed_hcl))
            except Exception as e:
                logger.debug(f"HCL2 parsing failed for {file_path}, falling back to regex: {e}")
                result.update(self._parse_with_regex(content))
            
            # Additional analysis
            result["analysis"].update(self._analyze_advanced_features(content))
            result["analysis"]["complexity_score"] = self._calculate_complexity_score(content)
            result["analysis"]["resource_count"] = len(result["resources"])
            
        except Exception as e:
            logger.error(f"Failed to parse Terraform file {file_path}: {e}")
        
        return result
    
    def _parse_hcl2_structure(self, parsed_hcl: Dict) -> Dict:
        """Parse HCL2 structure into standardized format"""
        result = {
            "resources": [],
            "variables": [],
            "locals": [],
            "outputs": [],
            "modules": [],
            "data_sources": [],
            "provider_config": []
        }
        
        # Parse resources
        for resource_type, resources in parsed_hcl.get("resource", {}).items():
            for resource_name, resource_config in resources.items():
                result["resources"].append({
                    "type": resource_type,
                    "name": resource_name,
                    "config": resource_config,
                    "provider": self._extract_provider_from_type(resource_type)
                })
        
        # Parse variables
        for var_name, var_config in parsed_hcl.get("variable", {}).items():
            result["variables"].append({
                "name": var_name,
                "config": var_config
            })
        
        # Parse locals
        for locals_block in parsed_hcl.get("locals", []):
            result["locals"].extend([
                {"name": k, "value": v} for k, v in locals_block.items()
            ])
        
        # Parse outputs
        for output_name, output_config in parsed_hcl.get("output", {}).items():
            result["outputs"].append({
                "name": output_name,
                "config": output_config
            })
        
        # Parse modules
        for module_name, module_config in parsed_hcl.get("module", {}).items():
            result["modules"].append({
                "name": module_name,
                "config": module_config
            })
        
        # Parse data sources
        for data_type, data_sources in parsed_hcl.get("data", {}).items():
            for data_name, data_config in data_sources.items():
                result["data_sources"].append({
                    "type": data_type,
                    "name": data_name,
                    "config": data_config
                })
        
        # Parse provider configurations
        for provider_name, provider_configs in parsed_hcl.get("provider", {}).items():
            if isinstance(provider_configs, list):
                for config_block in provider_configs:
                    result["provider_config"].append({
                        "provider": provider_name,
                        "config": config_block
                    })
            else:
                result["provider_config"].append({
                    "provider": provider_name,
                    "config": provider_configs
                })
        
        return result
    
    def _parse_with_regex(self, content: str) -> Dict:
        """Fallback regex-based parsing"""
        result = {
            "resources": [],
            "variables": [],
            "locals": [],
            "outputs": [],
            "modules": [],
            "data_sources": [],
            "provider_config": []
        }
        
        # Parse resources
        aws_resources = self.aws_resource_pattern.findall(content)
        azure_resources = self.azure_resource_pattern.findall(content)
        
        for resource_type, resource_name in aws_resources + azure_resources:
            result["resources"].append({
                "type": resource_type,
                "name": resource_name,
                "config": {},  # Limited config extraction with regex
                "provider": self._extract_provider_from_type(resource_type)
            })
        
        # Parse variables (simplified)
        variables = self.variable_pattern.findall(content)
        for var_name, var_block in variables:
            result["variables"].append({
                "name": var_name,
                "config": {"raw": var_block.strip()}
            })
        
        # Parse modules (simplified)
        modules = self.module_pattern.findall(content)
        for module_name, module_block in modules:
            result["modules"].append({
                "name": module_name,
                "config": {"raw": module_block.strip()}
            })
        
        return result
    
    def _extract_provider_from_type(self, resource_type: str) -> str:
        """Extract provider name from resource type"""
        if resource_type.startswith("aws_"):
            return "aws"
        elif resource_type.startswith("azurerm_"):
            return "azurerm"
        elif resource_type.startswith("google_"):
            return "google"
        else:
            return "unknown"
    
    def _analyze_advanced_features(self, content: str) -> Dict[str, bool]:
        """Analyze advanced Terraform features"""
        return {
            "has_dynamic_blocks": "dynamic" in content,
            "has_count": re.search(r'count\s*=', content) is not None,
            "has_for_each": re.search(r'for_each\s*=', content) is not None,
            "has_conditionals": "?" in content and ":" in content,
            "has_functions": any(func in content for func in [
                "length(", "keys(", "values(", "lookup(", "merge(", "join(",
                "split(", "replace(", "substr(", "format("
            ]),
            "has_interpolation": "${" in content,
            "has_terraform_blocks": "terraform {" in content,
            "has_required_providers": "required_providers" in content
        }
    
    def _calculate_complexity_score(self, content: str) -> int:
        """Calculate complexity score based on various factors"""
        score = 0
        
        # Basic constructs
        score += len(re.findall(r'resource\s+"[^"]+"\s+"[^"]+"', content)) * 2
        score += len(re.findall(r'data\s+"[^"]+"\s+"[^"]+"', content)) * 1
        score += len(re.findall(r'module\s+"[^"]+"', content)) * 3
        score += len(re.findall(r'variable\s+"[^"]+"', content)) * 1
        score += len(re.findall(r'locals\s*{', content)) * 2
        score += len(re.findall(r'output\s+"[^"]+"', content)) * 1
        
        # Advanced constructs
        score += content.count("dynamic") * 4
        score += len(re.findall(r'count\s*=', content)) * 3
        score += len(re.findall(r'for_each\s*=', content)) * 4
        score += content.count("${") * 1  # Interpolations
        
        # Functions usage
        tf_functions = [
            "length", "keys", "values", "lookup", "merge", "join", "split",
            "replace", "substr", "format", "flatten", "distinct", "sort"
        ]
        for func in tf_functions:
            score += content.count(f"{func}(") * 2
        
        return score
    
    def analyze_terraform_directory(self, directory: str) -> Dict[str, Any]:
        """Analyze entire Terraform directory"""
        dir_path = Path(directory)
        
        analysis = {
            "directory": str(dir_path),
            "files": {},
            "summary": {
                "total_files": 0,
                "total_resources": 0,
                "aws_resources": set(),
                "azure_resources": set(),
                "providers": set(),
                "total_complexity": 0,
                "has_backend_config": False,
                "has_version_constraints": False
            }
        }
        
        # Find all Terraform files
        tf_files = list(dir_path.glob("*.tf"))
        tfvars_files = list(dir_path.glob("*.tfvars"))
        
        # Analyze .tf files
        for tf_file in tf_files:
            file_analysis = self.parse_terraform_file(str(tf_file))
            analysis["files"][tf_file.name] = file_analysis
            
            # Update summary
            analysis["summary"]["total_files"] += 1
            analysis["summary"]["total_resources"] += len(file_analysis["resources"])
            analysis["summary"]["total_complexity"] += file_analysis["analysis"]["complexity_score"]
            
            for resource in file_analysis["resources"]:
                if resource["provider"] == "aws":
                    analysis["summary"]["aws_resources"].add(resource["type"])
                elif resource["provider"] == "azurerm":
                    analysis["summary"]["azure_resources"].add(resource["type"])
                analysis["summary"]["providers"].add(resource["provider"])
            
            # Check for backend and version configurations
            file_content = tf_file.read_text(encoding='utf-8', errors='ignore')
            if re.search(r'backend\s*"[^"]+"', file_content):
                analysis["summary"]["has_backend_config"] = True
            if "required_version" in file_content:
                analysis["summary"]["has_version_constraints"] = True
        
        # Analyze .tfvars files
        tfvars_data = {}
        for tfvars_file in tfvars_files:
            try:
                content = tfvars_file.read_text(encoding='utf-8', errors='ignore')
                variables = parse_terraform_variables(content)
                tfvars_data[tfvars_file.name] = variables
            except Exception as e:
                logger.warning(f"Error parsing tfvars file {tfvars_file}: {e}")
        
        analysis["tfvars"] = tfvars_data
        
        # Convert sets to lists for JSON serialization
        analysis["summary"]["aws_resources"] = list(analysis["summary"]["aws_resources"])
        analysis["summary"]["azure_resources"] = list(analysis["summary"]["azure_resources"])
        analysis["summary"]["providers"] = list(analysis["summary"]["providers"])
        
        return analysis
    
    def extract_resource_dependencies(self, resources: List[Dict]) -> Dict[str, List[str]]:
        """Extract dependencies between resources"""
        dependencies = {}
        
        for resource in resources:
            resource_id = f"{resource['type']}.{resource['name']}"
            deps = []
            
            # Look for references in resource configuration
            config_str = json.dumps(resource.get("config", {}))
            
            # Find references to other resources
            for other_resource in resources:
                other_id = f"{other_resource['type']}.{other_resource['name']}"
                if other_id != resource_id and other_id in config_str:
                    deps.append(other_id)
            
            dependencies[resource_id] = deps
        
        return dependencies
    
    def validate_terraform_syntax(self, content: str) -> Tuple[bool, List[str]]:
        """Validate Terraform syntax and return errors"""
        errors = []
        
        try:
            # Try to parse with HCL2
            hcl2.loads(content)
            return True, []
        except Exception as e:
            errors.append(f"HCL2 parsing error: {str(e)}")
        
        # Basic syntax validation with regex
        validation_rules = [
            (r'resource\s+"[^"]*"\s+"[^"]*"\s*\{', "Resource block syntax"),
            (r'variable\s+"[^"]*"\s*\{', "Variable block syntax"),
            (r'output\s+"[^"]*"\s*\{', "Output block syntax"),
        ]
        
        for pattern, rule_name in validation_rules:
            if not re.search(pattern, content):
                continue  # Rule not applicable
        
        # Check for common syntax errors
        brace_count = content.count('{') - content.count('}')
        if brace_count != 0:
            errors.append(f"Unmatched braces: {brace_count} difference")
        
        bracket_count = content.count('[') - content.count(']')
        if bracket_count != 0:
            errors.append(f"Unmatched brackets: {bracket_count} difference")
        
        return len(errors) == 0, errors
    
    def extract_terraform_configuration(self, directory: str) -> Dict[str, Any]:
        """Extract complete Terraform configuration from directory"""
        config_data = {
            "input_content": "",
            "structured_config": {},
            "variable_values": {},
            "file_contents": {},
            "analysis": {}
        }
        
        try:
            dir_path = Path(directory)
            
            # Collect all file contents
            content_parts = []
            
            # Process .tf files
            tf_files = list(dir_path.glob("*.tf"))
            for tf_file in sorted(tf_files):
                try:
                    content = tf_file.read_text(encoding='utf-8', errors='ignore')
                    cleaned = clean_terraform_content(content)
                    
                    content_parts.append(f"# File: {tf_file.name}")
                    content_parts.append(cleaned)
                    content_parts.append("")  # Empty line separator
                    
                    config_data["file_contents"][tf_file.name] = {
                        "raw": content,
                        "cleaned": cleaned,
                        "analysis": self.parse_terraform_file(str(tf_file))
                    }
                    
                except Exception as e:
                    logger.warning(f"Error reading {tf_file}: {e}")
            
            # Process .tfvars files
            tfvars_files = list(dir_path.glob("*.tfvars"))
            for tfvars_file in sorted(tfvars_files):
                try:
                    content = tfvars_file.read_text(encoding='utf-8', errors='ignore')
                    cleaned = clean_terraform_content(content)
                    
                    content_parts.append(f"# Variables: {tfvars_file.name}")
                    content_parts.append(cleaned)
                    content_parts.append("")
                    
                    # Parse variable values
                    variables = parse_terraform_variables(content)
                    config_data["variable_values"].update(variables)
                    
                except Exception as e:
                    logger.warning(f"Error reading {tfvars_file}: {e}")
            
            # Create combined input content
            config_data["input_content"] = "\n".join(content_parts).strip()
            
            # Perform directory-level analysis
            config_data["analysis"] = self.analyze_terraform_directory(directory)
            
            # Create structured configuration
            config_data["structured_config"] = self._create_structured_config(config_data)
            
        except Exception as e:
            logger.error(f"Error extracting configuration from {directory}: {e}")
        
        return config_data
    
    def _create_structured_config(self, config_data: Dict) -> Dict:
        """Create structured configuration summary"""
        structured = {
            "providers": [],
            "resources_by_provider": {},
            "variables": {},
            "modules": [],
            "complexity_metrics": {}
        }
        
        try:
            # Aggregate from all files
            all_resources = []
            all_variables = []
            all_modules = []
            
            for file_name, file_data in config_data["file_contents"].items():
                file_analysis = file_data.get("analysis", {})
                
                all_resources.extend(file_analysis.get("resources", []))
                all_variables.extend(file_analysis.get("variables", []))
                all_modules.extend(file_analysis.get("modules", []))
            
            # Group resources by provider
            for resource in all_resources:
                provider = resource.get("provider", "unknown")
                if provider not in structured["resources_by_provider"]:
                    structured["resources_by_provider"][provider] = []
                structured["resources_by_provider"][provider].append(resource)
                
                if provider not in structured["providers"]:
                    structured["providers"].append(provider)
            
            # Collect variables
            for variable in all_variables:
                structured["variables"][variable["name"]] = variable["config"]
            
            # Add tfvars values
            structured["variables"].update(config_data.get("variable_values", {}))
            
            # Collect modules
            structured["modules"] = [
                {"name": module["name"], "source": module.get("config", {}).get("source", "")}
                for module in all_modules
            ]
            
            # Calculate complexity metrics
            total_complexity = sum(
                file_data.get("analysis", {}).get("analysis", {}).get("complexity_score", 0)
                for file_data in config_data["file_contents"].values()
            )
            
            structured["complexity_metrics"] = {
                "total_complexity": total_complexity,
                "resource_count": len(all_resources),
                "provider_count": len(structured["providers"]),
                "variable_count": len(all_variables),
                "module_count": len(all_modules)
            }
            
        except Exception as e:
            logger.error(f"Error creating structured config: {e}")
        
        return structured
    
    def detect_infrastructure_patterns(self, analysis: Dict) -> List[str]:
        """Detect common infrastructure patterns"""
        patterns = []
        
        resources_by_provider = analysis.get("structured_config", {}).get("resources_by_provider", {})
        
        # AWS patterns
        aws_resources = [r["type"] for r in resources_by_provider.get("aws", [])]
        if "aws_vpc" in aws_resources and "aws_subnet" in aws_resources:
            patterns.append("vpc_networking")
        if "aws_instance" in aws_resources and "aws_security_group" in aws_resources:
            patterns.append("ec2_deployment")
        if "aws_s3_bucket" in aws_resources and "aws_cloudfront_distribution" in aws_resources:
            patterns.append("static_website")
        if "aws_ecs_cluster" in aws_resources or "aws_eks_cluster" in aws_resources:
            patterns.append("container_orchestration")
        if "aws_rds_instance" in aws_resources or "aws_db_instance" in aws_resources:
            patterns.append("database_deployment")
        
        # Azure patterns
        azure_resources = [r["type"] for r in resources_by_provider.get("azurerm", [])]
        if "azurerm_virtual_network" in azure_resources and "azurerm_subnet" in azure_resources:
            patterns.append("azure_networking")
        if "azurerm_virtual_machine" in azure_resources:
            patterns.append("azure_vm_deployment")
        if "azurerm_app_service" in azure_resources:
            patterns.append("azure_web_app")
        if "azurerm_kubernetes_cluster" in azure_resources:
            patterns.append("azure_aks")
        
        return patterns

def main():
    """Test the analyzer"""
    analyzer = TerraformAnalyzer()
    
    # Test with a sample directory
    test_dir = "/path/to/terraform/directory"
    if os.path.exists(test_dir):
        result = analyzer.extract_terraform_configuration(test_dir)
        print(json.dumps(result, indent=2, default=str))
    else:
        print("Please provide a valid Terraform directory for testing")

if __name__ == "__main__":
    main()
