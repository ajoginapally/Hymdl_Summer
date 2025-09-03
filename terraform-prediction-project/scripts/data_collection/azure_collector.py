"""
Azure-specific repository collector and analyzer
"""

import os
import json
import time
import requests
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import (
    retry_on_failure, safe_file_operation, ProgressTracker,
    extract_resources_from_terraform, cache
)
from .terraform_analyzer import TerraformAnalyzer

logger = logging.getLogger(__name__)

class AzureTerraformCollector:
    """Specialized collector for Azure Terraform repositories"""
    
    def __init__(self):
        self.analyzer = TerraformAnalyzer()
        self.session = requests.Session()
        if config.github_token:
            self.session.headers.update({"Authorization": f"token {config.github_token}"})
        
        # Azure-specific search patterns
        self.azure_search_queries = [
            "terraform azurerm language:HCL stars:>5",
            "terraform azure resource group language:HCL",
            "terraform azure virtual machine language:HCL",
            "terraform azure app service language:HCL",
            "terraform azure storage account language:HCL",
            "terraform azure sql database language:HCL",
            "terraform azure kubernetes language:HCL",
            "infrastructure as code azure terraform language:HCL",
            "azure terraform examples language:HCL",
            "azurerm provider terraform language:HCL"
        ]
        
        # Common Azure resource patterns to prioritize
        self.priority_azure_patterns = [
            ["azurerm_resource_group", "azurerm_virtual_network"],
            ["azurerm_virtual_machine", "azurerm_network_interface"],
            ["azurerm_app_service", "azurerm_app_service_plan"],
            ["azurerm_storage_account", "azurerm_storage_blob"],
            ["azurerm_sql_server", "azurerm_sql_database"],
            ["azurerm_kubernetes_cluster", "azurerm_kubernetes_cluster_node_pool"]
        ]
    
    def search_azure_repositories(self) -> List[Dict]:
        """Search specifically for Azure Terraform repositories"""
        logger.info("Searching for Azure Terraform repositories")
        
        all_repos = []
        
        for query in self.azure_search_queries:
            try:
                cache_key = f"azure_search_{hash(query)}"
                cached_repos = cache.get(cache_key)
                
                if cached_repos and cache.is_valid(cache_key):
                    all_repos.extend(cached_repos)
                    continue
                
                repos = self._search_github_api(query)
                cache.set(cache_key, repos, expiry_hours=12)
                all_repos.extend(repos)
                
                time.sleep(2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Failed to search with query '{query}': {e}")
                continue
        
        # Remove duplicates and filter for Azure content
        unique_repos = self._filter_and_deduplicate_azure_repos(all_repos)
        
        logger.info(f"Found {len(unique_repos)} unique Azure repositories")
        return unique_repos
    
    def _search_github_api(self, query: str, per_page: int = 50, max_pages: int = 10) -> List[Dict]:
        """Search GitHub API with pagination"""
        repos = []
        
        for page in range(1, max_pages + 1):
            try:
                url = f"{config.github_api_base_url}/search/repositories"
                params = {
                    "q": query,
                    "sort": "stars",
                    "order": "desc",
                    "per_page": per_page,
                    "page": page
                }
                
                response = self.session.get(url, params=params)
                response.raise_for_status()
                
                data = response.json()
                page_repos = data.get("items", [])
                
                if not page_repos:
                    break
                
                repos.extend(page_repos)
                time.sleep(1)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                logger.error(f"API request failed for query '{query}', page {page}: {e}")
                break
        
        return repos
    
    def _filter_and_deduplicate_azure_repos(self, repos: List[Dict]) -> List[Dict]:
        """Filter and deduplicate Azure repositories"""
        seen_ids = set()
        filtered_repos = []
        
        for repo in repos:
            repo_id = repo.get("id")
            if not repo_id or repo_id in seen_ids:
                continue
            
            # Filter criteria for Azure repos
            description = (repo.get("description", "") or "").lower()
            name = repo.get("name", "").lower()
            
            # Must contain Azure-related terms
            azure_keywords = ["azure", "azurerm", "az", "microsoft"]
            terraform_keywords = ["terraform", "infrastructure", "iac"]
            
            has_azure = any(keyword in description + name for keyword in azure_keywords)
            has_terraform = any(keyword in description + name for keyword in terraform_keywords)
            
            if has_azure and has_terraform:
                seen_ids.add(repo_id)
                filtered_repos.append(repo)
        
        return filtered_repos
    
    def analyze_azure_repository(self, repo_data: Dict) -> Optional[Dict]:
        """Analyze Azure repository for content quality"""
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                repo_dir = os.path.join(temp_dir, "repo")
                
                # Clone repository
                clone_success = self._clone_repository(repo_data, repo_dir)
                if not clone_success:
                    return None
                
                # Analyze Terraform content
                analysis = self._analyze_azure_terraform_content(repo_dir)
                
                # Filter based on Azure content quality
                if not self._meets_azure_quality_criteria(analysis):
                    return None
                
                return {
                    **repo_data,
                    "azure_analysis": analysis,
                    "collection_timestamp": datetime.now().isoformat(),
                    "azure_quality_score": self._calculate_azure_quality_score(analysis)
                }
                
        except Exception as e:
            logger.warning(f"Failed to analyze Azure repository {repo_data.get('full_name')}: {e}")
            return None
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def _clone_repository(self, repo_data: Dict, target_dir: str) -> bool:
        """Clone repository with retry logic"""
        try:
            import git
            git.Repo.clone_from(
                repo_data["clone_url"],
                target_dir,
                depth=1,
                timeout=300
            )
            return True
        except Exception as e:
            logger.warning(f"Failed to clone {repo_data['full_name']}: {e}")
            return False
    
    def _analyze_azure_terraform_content(self, repo_dir: str) -> Dict:
        """Analyze Azure Terraform content in repository"""
        analysis = {
            "terraform_directories": [],
            "total_azure_resources": 0,
            "azure_services": set(),
            "resource_patterns": [],
            "complexity_metrics": {},
            "file_quality": {}
        }
        
        repo_path = Path(repo_dir)
        
        # Find all directories with Terraform files
        tf_dirs = []
        for root, dirs, files in os.walk(repo_path):
            if any(f.endswith('.tf') for f in files):
                tf_dirs.append(root)
        
        for tf_dir in tf_dirs:
            try:
                dir_analysis = self.analyzer.analyze_terraform_directory(tf_dir)
                
                azure_resources = dir_analysis["summary"].get("azure_resources", [])
                if azure_resources:  # Only include directories with Azure resources
                    relative_path = str(Path(tf_dir).relative_to(repo_path))
                    
                    analysis["terraform_directories"].append({
                        "path": relative_path,
                        "azure_resources": azure_resources,
                        "total_resources": dir_analysis["summary"]["total_resources"],
                        "complexity": dir_analysis["summary"]["total_complexity"]
                    })
                    
                    analysis["total_azure_resources"] += len(azure_resources)
                    analysis["azure_services"].update(azure_resources)
                    
                    # Detect patterns in this directory
                    patterns = self._detect_azure_patterns(azure_resources)
                    analysis["resource_patterns"].extend(patterns)
                    
            except Exception as e:
                logger.warning(f"Error analyzing directory {tf_dir}: {e}")
        
        # Convert sets to lists for JSON serialization
        analysis["azure_services"] = list(analysis["azure_services"])
        analysis["resource_patterns"] = list(set(analysis["resource_patterns"]))  # Remove duplicates
        
        # Calculate overall complexity metrics
        analysis["complexity_metrics"] = {
            "avg_complexity_per_dir": (
                sum(d["complexity"] for d in analysis["terraform_directories"]) / 
                len(analysis["terraform_directories"])
            ) if analysis["terraform_directories"] else 0,
            "total_directories": len(analysis["terraform_directories"]),
            "azure_service_coverage": len(set(analysis["azure_services"]) & set(config.azure_services))
        }
        
        return analysis
    
    def _detect_azure_patterns(self, azure_resources: List[str]) -> List[str]:
        """Detect Azure infrastructure patterns"""
        patterns = []
        
        for pattern_resources in self.priority_azure_patterns:
            if all(resource in azure_resources for resource in pattern_resources):
                pattern_name = "_".join(r.replace("azurerm_", "") for r in pattern_resources)
                patterns.append(pattern_name)
        
        # Additional pattern detection
        if "azurerm_resource_group" in azure_resources:
            patterns.append("basic_azure_setup")
        
        if any("kubernetes" in resource for resource in azure_resources):
            patterns.append("kubernetes_deployment")
        
        if any("app_service" in resource for resource in azure_resources):
            patterns.append("web_application")
        
        if any("sql" in resource for resource in azure_resources):
            patterns.append("database_deployment")
        
        return patterns
    
    def _meets_azure_quality_criteria(self, analysis: Dict) -> bool:
        """Check if Azure repository meets quality criteria"""
        # Must have at least one directory with Azure resources
        if not analysis["terraform_directories"]:
            return False
        
        # Must have reasonable number of Azure resources
        if analysis["total_azure_resources"] < 2:
            return False
        
        # Should cover multiple Azure services
        if len(analysis["azure_services"]) < 2:
            return False
        
        # Should have some complexity (not just trivial examples)
        avg_complexity = analysis["complexity_metrics"]["avg_complexity_per_dir"]
        if avg_complexity < 5:
            return False
        
        return True
    
    def _calculate_azure_quality_score(self, analysis: Dict) -> float:
        """Calculate quality score for Azure repository"""
        score = 0.0
        
        # Service coverage (0-50 points)
        azure_coverage = len(set(analysis["azure_services"]) & set(config.azure_services))
        score += min(50, azure_coverage * 5)
        
        # Resource quantity (0-30 points)
        resource_score = min(30, analysis["total_azure_resources"] * 2)
        score += resource_score
        
        # Pattern complexity (0-20 points)
        pattern_score = min(20, len(analysis["resource_patterns"]) * 4)
        score += pattern_score
        
        return score
    
    def collect_azure_repositories(self, max_repos: int = 50) -> List[Dict]:
        """Main method to collect Azure repositories"""
        logger.info("Starting Azure repository collection")
        
        # Search for repositories
        repos = self.search_azure_repositories()
        
        # Analyze repositories in parallel
        analyzed_repos = []
        progress = ProgressTracker(min(len(repos), max_repos), "Azure Repository Analysis")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []
            for repo in repos[:max_repos]:
                future = executor.submit(self.analyze_azure_repository, repo)
                futures.append(future)
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        analyzed_repos.append(result)
                    progress.update()
                except Exception as e:
                    logger.error(f"Azure repository analysis failed: {e}")
                    progress.update()
        
        progress.finish()
        
        # Sort by Azure quality score
        analyzed_repos.sort(key=lambda x: x.get("azure_quality_score", 0), reverse=True)
        
        logger.info(f"Azure collection complete! Found {len(analyzed_repos)} quality repositories")
        return analyzed_repos
    
    def save_azure_repository_data(self, repos: List[Dict]):
        """Save Azure repository data"""
        output_dir = config.data_dir / "azure"
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed data
        output_file = output_dir / "azure_terraform_repositories.json"
        metadata = {
            "collection_date": datetime.now().isoformat(),
            "total_repositories": len(repos),
            "azure_specific": True,
            "repositories": repos
        }
        
        safe_file_operation("write", str(output_file), metadata)
        
        # Generate Azure-specific summary
        summary = self._generate_azure_summary(repos)
        summary_file = output_dir / "azure_collection_summary.json"
        safe_file_operation("write", str(summary_file), summary)
        
        logger.info(f"Saved {len(repos)} Azure repositories to {output_file}")
        return str(output_file)
    
    def _generate_azure_summary(self, repos: List[Dict]) -> Dict:
        """Generate Azure-specific collection summary"""
        azure_services_found = set()
        pattern_distribution = {}
        total_azure_resources = 0
        
        for repo in repos:
            azure_analysis = repo.get("azure_analysis", {})\n            \n            azure_services_found.update(azure_analysis.get("azure_services", []))\n            total_azure_resources += azure_analysis.get("total_azure_resources", 0)\n            \n            for pattern in azure_analysis.get("resource_patterns", []):\n                pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + 1\n        \n        return {\n            "total_repositories": len(repos),\n            "total_azure_resources": total_azure_resources,\n            "azure_services_coverage": {\n                "found": len(azure_services_found),\n                "target": len(config.azure_services),\n                "percentage": (len(azure_services_found) / len(config.azure_services)) * 100,\n                "services": sorted(list(azure_services_found))\n            },\n            "pattern_distribution": pattern_distribution,\n            "avg_quality_score": sum(r.get("azure_quality_score", 0) for r in repos) / len(repos) if repos else 0\n        }\n\ndef main():\n    \"\"\"Main execution for Azure collection\"\"\"\n    collector = AzureTerraformCollector()\n    \n    try:\n        repos = collector.collect_azure_repositories(max_repos=30)\n        output_file = collector.save_azure_repository_data(repos)\n        \n        print(f\"Azure repository collection completed!\")\n        print(f"Results saved to: {output_file}\")\n        \n    except Exception as e:\n        logger.error(f\"Azure collection failed: {e}\")\n        sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()"}}
</function_calls>

<function_results>{"file_content_after_create":{"path":"/Users/arnavj/Work/coding/ai_work/hymdl/model/terraform-prediction-project/scripts/data_collection/azure_collector.py","content":"\"\"\"Azure-specific repository collector and analyzer\"\"\"\n\nimport os\nimport json\nimport time\nimport requests\nimport tempfile\nimport concurrent.futures\nfrom pathlib import Path\nfrom typing import Dict, List, Optional\nfrom datetime import datetime\nimport logging\n\n# Add project root to path\nimport sys\nsys.path.append(str(Path(__file__).parent.parent.parent))\n\nfrom config.config import config\nfrom config.utils import (\n    retry_on_failure, safe_file_operation, ProgressTracker,\n    extract_resources_from_terraform, cache\n)\nfrom .terraform_analyzer import TerraformAnalyzer\n\nlogger = logging.getLogger(__name__)\n\nclass AzureTerraformCollector:\n    \"\"\"Specialized collector for Azure Terraform repositories\"\"\"\n    \n    def __init__(self):\n        self.analyzer = TerraformAnalyzer()\n        self.session = requests.Session()\n        if config.github_token:\n            self.session.headers.update({\"Authorization\": f\"token {config.github_token}\"})\n        \n        # Azure-specific search patterns\n        self.azure_search_queries = [\n            \"terraform azurerm language:HCL stars:>5\",\n            \"terraform azure resource group language:HCL\",\n            \"terraform azure virtual machine language:HCL\",\n            \"terraform azure app service language:HCL\",\n            \"terraform azure storage account language:HCL\",\n            \"terraform azure sql database language:HCL\",\n            \"terraform azure kubernetes language:HCL\",\n            \"infrastructure as code azure terraform language:HCL\",\n            \"azure terraform examples language:HCL\",\n            \"azurerm provider terraform language:HCL\"\n        ]\n        \n        # Common Azure resource patterns to prioritize\n        self.priority_azure_patterns = [\n            [\"azurerm_resource_group\", \"azurerm_virtual_network\"],\n            [\"azurerm_virtual_machine\", \"azurerm_network_interface\"],\n            [\"azurerm_app_service\", \"azurerm_app_service_plan\"],\n            [\"azurerm_storage_account\", \"azurerm_storage_blob\"],\n            [\"azurerm_sql_server\", \"azurerm_sql_database\"],\n            [\"azurerm_kubernetes_cluster\", \"azurerm_kubernetes_cluster_node_pool\"]\n        ]\n    \n    def search_azure_repositories(self) -> List[Dict]:\n        \"\"\"Search specifically for Azure Terraform repositories\"\"\"\n        logger.info(\"Searching for Azure Terraform repositories\")\n        \n        all_repos = []\n        \n        for query in self.azure_search_queries:\n            try:\n                cache_key = f\"azure_search_{hash(query)}\"\n                cached_repos = cache.get(cache_key)\n                \n                if cached_repos and cache.is_valid(cache_key):\n                    all_repos.extend(cached_repos)\n                    continue\n                \n                repos = self._search_github_api(query)\n                cache.set(cache_key, repos, expiry_hours=12)\n                all_repos.extend(repos)\n                \n                time.sleep(2)  # Rate limiting\n                \n            except Exception as e:\n                logger.error(f\"Failed to search with query '{query}': {e}\")\n                continue\n        \n        # Remove duplicates and filter for Azure content\n        unique_repos = self._filter_and_deduplicate_azure_repos(all_repos)\n        \n        logger.info(f\"Found {len(unique_repos)} unique Azure repositories\")\n        return unique_repos\n    \n    def _search_github_api(self, query: str, per_page: int = 50, max_pages: int = 10) -> List[Dict]:\n        \"\"\"Search GitHub API with pagination\"\"\"\n        repos = []\n        \n        for page in range(1, max_pages + 1):\n            try:\n                url = f\"{config.github_api_base_url}/search/repositories\"\n                params = {\n                    \"q\": query,\n                    \"sort\": \"stars\",\n                    \"order\": \"desc\",\n                    \"per_page\": per_page,\n                    \"page\": page\n                }\n                \n                response = self.session.get(url, params=params)\n                response.raise_for_status()\n                \n                data = response.json()\n                page_repos = data.get(\"items\", [])\n                \n                if not page_repos:\n                    break\n                \n                repos.extend(page_repos)\n                time.sleep(1)  # Rate limiting\n                \n            except requests.exceptions.RequestException as e:\n                logger.error(f\"API request failed for query '{query}', page {page}: {e}\")\n                break\n        \n        return repos\n    \n    def _filter_and_deduplicate_azure_repos(self, repos: List[Dict]) -> List[Dict]:\n        \"\"\"Filter and deduplicate Azure repositories\"\"\"\n        seen_ids = set()\n        filtered_repos = []\n        \n        for repo in repos:\n            repo_id = repo.get(\"id\")\n            if not repo_id or repo_id in seen_ids:\n                continue\n            \n            # Filter criteria for Azure repos\n            description = (repo.get(\"description\", \"\") or \"\").lower()\n            name = repo.get(\"name\", \"\").lower()\n            \n            # Must contain Azure-related terms\n            azure_keywords = [\"azure\", \"azurerm\", \"az\", \"microsoft\"]\n            terraform_keywords = [\"terraform\", \"infrastructure\", \"iac\"]\n            \n            has_azure = any(keyword in description + name for keyword in azure_keywords)\n            has_terraform = any(keyword in description + name for keyword in terraform_keywords)\n            \n            if has_azure and has_terraform:\n                seen_ids.add(repo_id)\n                filtered_repos.append(repo)\n        \n        return filtered_repos\n    \n    def analyze_azure_repository(self, repo_data: Dict) -> Optional[Dict]:\n        \"\"\"Analyze Azure repository for content quality\"\"\"\n        try:\n            with tempfile.TemporaryDirectory() as temp_dir:\n                repo_dir = os.path.join(temp_dir, \"repo\")\n                \n                # Clone repository\n                clone_success = self._clone_repository(repo_data, repo_dir)\n                if not clone_success:\n                    return None\n                \n                # Analyze Terraform content\n                analysis = self._analyze_azure_terraform_content(repo_dir)\n                \n                # Filter based on Azure content quality\n                if not self._meets_azure_quality_criteria(analysis):\n                    return None\n                \n                return {\n                    **repo_data,\n                    \"azure_analysis\": analysis,\n                    \"collection_timestamp\": datetime.now().isoformat(),\n                    \"azure_quality_score\": self._calculate_azure_quality_score(analysis)\n                }\n                \n        except Exception as e:\n            logger.warning(f\"Failed to analyze Azure repository {repo_data.get('full_name')}: {e}\")\n            return None\n    \n    @retry_on_failure(max_retries=3, delay=2.0)\n    def _clone_repository(self, repo_data: Dict, target_dir: str) -> bool:\n        \"\"\"Clone repository with retry logic\"\"\"\n        try:\n            import git\n            git.Repo.clone_from(\n                repo_data[\"clone_url\"],\n                target_dir,\n                depth=1,\n                timeout=300\n            )\n            return True\n        except Exception as e:\n            logger.warning(f\"Failed to clone {repo_data['full_name']}: {e}\")\n            return False\n    \n    def _analyze_azure_terraform_content(self, repo_dir: str) -> Dict:\n        \"\"\"Analyze Azure Terraform content in repository\"\"\"\n        analysis = {\n            \"terraform_directories\": [],\n            \"total_azure_resources\": 0,\n            \"azure_services\": set(),\n            \"resource_patterns\": [],\n            \"complexity_metrics\": {},\n            \"file_quality\": {}\n        }\n        \n        repo_path = Path(repo_dir)\n        \n        # Find all directories with Terraform files\n        tf_dirs = []\n        for root, dirs, files in os.walk(repo_path):\n            if any(f.endswith('.tf') for f in files):\n                tf_dirs.append(root)\n        \n        for tf_dir in tf_dirs:\n            try:\n                dir_analysis = self.analyzer.analyze_terraform_directory(tf_dir)\n                \n                azure_resources = dir_analysis[\"summary\"].get(\"azure_resources\", [])\n                if azure_resources:  # Only include directories with Azure resources\n                    relative_path = str(Path(tf_dir).relative_to(repo_path))\n                    \n                    analysis[\"terraform_directories\"].append({\n                        \"path\": relative_path,\n                        \"azure_resources\": azure_resources,\n                        \"total_resources\": dir_analysis[\"summary\"][\"total_resources\"],\n                        \"complexity\": dir_analysis[\"summary\"][\"total_complexity\"]\n                    })\n                    \n                    analysis[\"total_azure_resources\"] += len(azure_resources)\n                    analysis[\"azure_services\"].update(azure_resources)\n                    \n                    # Detect patterns in this directory\n                    patterns = self._detect_azure_patterns(azure_resources)\n                    analysis[\"resource_patterns\"].extend(patterns)\n                    \n            except Exception as e:\n                logger.warning(f\"Error analyzing directory {tf_dir}: {e}\")\n        \n        # Convert sets to lists for JSON serialization\n        analysis[\"azure_services\"] = list(analysis[\"azure_services\"])\n        analysis[\"resource_patterns\"] = list(set(analysis[\"resource_patterns\"]))  # Remove duplicates\n        \n        # Calculate overall complexity metrics\n        analysis[\"complexity_metrics\"] = {\n            \"avg_complexity_per_dir\": (\n                sum(d[\"complexity\"] for d in analysis[\"terraform_directories\"]) / \n                len(analysis[\"terraform_directories\"])\n            ) if analysis[\"terraform_directories\"] else 0,\n            \"total_directories\": len(analysis[\"terraform_directories\"]),\n            \"azure_service_coverage\": len(set(analysis[\"azure_services\"]) & set(config.azure_services))\n        }\n        \n        return analysis\n    \n    def _detect_azure_patterns(self, azure_resources: List[str]) -> List[str]:\n        \"\"\"Detect Azure infrastructure patterns\"\"\"\n        patterns = []\n        \n        for pattern_resources in self.priority_azure_patterns:\n            if all(resource in azure_resources for resource in pattern_resources):\n                pattern_name = \"_\".join(r.replace(\"azurerm_\", \"\") for r in pattern_resources)\n                patterns.append(pattern_name)\n        \n        # Additional pattern detection\n        if \"azurerm_resource_group\" in azure_resources:\n            patterns.append(\"basic_azure_setup\")\n        \n        if any(\"kubernetes\" in resource for resource in azure_resources):\n            patterns.append(\"kubernetes_deployment\")\n        \n        if any(\"app_service\" in resource for resource in azure_resources):\n            patterns.append(\"web_application\")\n        \n        if any(\"sql\" in resource for resource in azure_resources):\n            patterns.append(\"database_deployment\")\n        \n        return patterns\n    \n    def _meets_azure_quality_criteria(self, analysis: Dict) -> bool:\n        \"\"\"Check if Azure repository meets quality criteria\"\"\"\n        # Must have at least one directory with Azure resources\n        if not analysis[\"terraform_directories\"]:\n            return False\n        \n        # Must have reasonable number of Azure resources\n        if analysis[\"total_azure_resources\"] < 2:\n            return False\n        \n        # Should cover multiple Azure services\n        if len(analysis[\"azure_services\"]) < 2:\n            return False\n        \n        # Should have some complexity (not just trivial examples)\n        avg_complexity = analysis[\"complexity_metrics\"][\"avg_complexity_per_dir\"]\n        if avg_complexity < 5:\n            return False\n        \n        return True\n    \n    def _calculate_azure_quality_score(self, analysis: Dict) -> float:\n        \"\"\"Calculate quality score for Azure repository\"\"\"\n        score = 0.0\n        \n        # Service coverage (0-50 points)\n        azure_coverage = len(set(analysis[\"azure_services\"]) & set(config.azure_services))\n        score += min(50, azure_coverage * 5)\n        \n        # Resource quantity (0-30 points)\n        resource_score = min(30, analysis[\"total_azure_resources\"] * 2)\n        score += resource_score\n        \n        # Pattern complexity (0-20 points)\n        pattern_score = min(20, len(analysis[\"resource_patterns\"]) * 4)\n        score += pattern_score\n        \n        return score\n    \n    def collect_azure_repositories(self, max_repos: int = 50) -> List[Dict]:\n        \"\"\"Main method to collect Azure repositories\"\"\"\n        logger.info(\"Starting Azure repository collection\")\n        \n        # Search for repositories\n        repos = self.search_azure_repositories()\n        \n        # Analyze repositories in parallel\n        analyzed_repos = []\n        progress = ProgressTracker(min(len(repos), max_repos), \"Azure Repository Analysis\")\n        \n        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:\n            futures = []\n            for repo in repos[:max_repos]:\n                future = executor.submit(self.analyze_azure_repository, repo)\n                futures.append(future)\n            \n            for future in concurrent.futures.as_completed(futures):\n                try:\n                    result = future.result()\n                    if result:\n                        analyzed_repos.append(result)\n                    progress.update()\n                except Exception as e:\n                    logger.error(f\"Azure repository analysis failed: {e}\")\n                    progress.update()\n        \n        progress.finish()\n        \n        # Sort by Azure quality score\n        analyzed_repos.sort(key=lambda x: x.get(\"azure_quality_score\", 0), reverse=True)\n        \n        logger.info(f\"Azure collection complete! Found {len(analyzed_repos)} quality repositories\")\n        return analyzed_repos\n    \n    def save_azure_repository_data(self, repos: List[Dict]):\n        \"\"\"Save Azure repository data\"\"\"\n        output_dir = config.data_dir / \"azure\"\n        output_dir.mkdir(exist_ok=True)\n        \n        # Save detailed data\n        output_file = output_dir / \"azure_terraform_repositories.json\"\n        metadata = {\n            \"collection_date\": datetime.now().isoformat(),\n            \"total_repositories\": len(repos),\n            \"azure_specific\": True,\n            \"repositories\": repos\n        }\n        \n        safe_file_operation(\"write\", str(output_file), metadata)\n        \n        # Generate Azure-specific summary\n        summary = self._generate_azure_summary(repos)\n        summary_file = output_dir / \"azure_collection_summary.json\"\n        safe_file_operation(\"write\", str(summary_file), summary)\n        \n        logger.info(f\"Saved {len(repos)} Azure repositories to {output_file}\")\n        return str(output_file)\n    \n    def _generate_azure_summary(self, repos: List[Dict]) -> Dict:\n        \"\"\"Generate Azure-specific collection summary\"\"\"\n        azure_services_found = set()\n        pattern_distribution = {}\n        total_azure_resources = 0\n        \n        for repo in repos:\n            azure_analysis = repo.get(\"azure_analysis\", {})\n            \n            azure_services_found.update(azure_analysis.get(\"azure_services\", []))\n            total_azure_resources += azure_analysis.get(\"total_azure_resources\", 0)\n            \n            for pattern in azure_analysis.get(\"resource_patterns\", []):\n                pattern_distribution[pattern] = pattern_distribution.get(pattern, 0) + 1\n        \n        return {\n            \"total_repositories\": len(repos),\n            \"total_azure_resources\": total_azure_resources,\n            \"azure_services_coverage\": {\n                \"found\": len(azure_services_found),\n                \"target\": len(config.azure_services),\n                \"percentage\": (len(azure_services_found) / len(config.azure_services)) * 100,\n                \"services\": sorted(list(azure_services_found))\n            },\n            \"pattern_distribution\": pattern_distribution,\n            \"avg_quality_score\": sum(r.get(\"azure_quality_score\", 0) for r in repos) / len(repos) if repos else 0\n        }\n\ndef main():\n    \"\"\"Main execution for Azure collection\"\"\"\n    collector = AzureTerraformCollector()\n    \n    try:\n        repos = collector.collect_azure_repositories(max_repos=30)\n        output_file = collector.save_azure_repository_data(repos)\n        \n        print(f\"Azure repository collection completed!\")\n        print(f\"Results saved to: {output_file}\")\n        \n    except Exception as e:\n        logger.error(f\"Azure collection failed: {e}\")\n        sys.exit(1)\n\nif __name__ == \"__main__\":\n    main()\n"}}
</function_calls>

<function_calls>
<invoke name="mark_todo_as_done">
<parameter name="todo_ids">["960821c7-69f3-4a0b-8c51-459f1f81ec06"]
