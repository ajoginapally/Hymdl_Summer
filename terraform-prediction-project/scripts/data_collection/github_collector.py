"""
GitHub repository collector for Terraform projects
"""

import os
import json
import time
import requests
import tempfile
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime
import logging
from github import Github
import git

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from config.config import config
from config.utils import (
    retry_on_failure, safe_file_operation, ProgressTracker,
    extract_resources_from_terraform, sanitize_filename, cache
)

logger = logging.getLogger(__name__)

class GitHubTerraformCollector:
    """Collect Terraform repositories from GitHub with advanced filtering"""
    
    def __init__(self):
        self.github_client = Github(config.github_token) if config.github_token else None
        self.session = requests.Session()
        if config.github_token:
            self.session.headers.update({"Authorization": f"token {config.github_token}"})
        
        self.collected_repos: List[Dict] = []
        self.processed_repo_ids: Set[str] = set()
        
        # Create output directory
        self.output_dir = config.data_dir / "raw"
        self.output_dir.mkdir(exist_ok=True)
    
    @retry_on_failure(max_retries=3, delay=2.0)
    def search_repositories(self, query: str, per_page: int = 50, max_pages: int = 20) -> List[Dict]:
        """Search GitHub repositories with retry logic"""
        logger.info(f"Searching GitHub with query: {query}")
        
        # Check cache first
        cache_key = f"github_search_{hashlib.md5(query.encode()).hexdigest()}"
        cached_result = cache.get(cache_key)
        if cached_result and cache.is_valid(cache_key):
            logger.info(f"Using cached results for query: {query}")
            return cached_result
        
        repos = []
        for page in range(1, max_pages + 1):
            try:
                if self.github_client:
                    # Use PyGithub for more robust API access
                    search_result = self.github_client.search_repositories(
                        query=query,
                        sort="stars",
                        order="desc"
                    )
                    
                    # Convert to page-based access
                    start_idx = (page - 1) * per_page
                    end_idx = start_idx + per_page
                    
                    try:
                        page_repos = search_result[start_idx:end_idx]
                        if not page_repos:
                            break
                        
                        for repo in page_repos:
                            repos.append({
                                "id": repo.id,
                                "name": repo.name,
                                "full_name": repo.full_name,
                                "clone_url": repo.clone_url,
                                "stargazers_count": repo.stargazers_count,
                                "size": repo.size,
                                "language": repo.language,
                                "created_at": repo.created_at.isoformat() if repo.created_at else None,
                                "updated_at": repo.updated_at.isoformat() if repo.updated_at else None,
                                "description": repo.description,
                                "default_branch": repo.default_branch
                            })
                    except Exception as e:
                        logger.warning(f"Error accessing page {page}: {e}")
                        break
                        
                else:
                    # Fallback to REST API
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
                    repos.extend(data.get("items", []))
                    
                    if len(data.get("items", [])) < per_page:
                        break
                
                # Rate limiting
                time.sleep(1.2)  # Be conservative with rate limits
                
            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                time.sleep(5)  # Longer delay on error
                continue
        
        # Cache results
        cache.set(cache_key, repos, expiry_hours=6)
        
        logger.info(f"Found {len(repos)} repositories for query: {query}")
        return repos
    
    def get_comprehensive_search_queries(self) -> List[str]:
        """Generate comprehensive search queries for Terraform repositories"""
        base_queries = [
            "terraform aws language:HCL stars:>10 size:<50000",
            "terraform infrastructure language:HCL stars:>5 size:<100000", 
            "terraform modules aws language:HCL stars:>3",
            "terraform examples aws language:HCL",
            "terraform azurerm language:HCL stars:>5",
            "terraform azure language:HCL stars:>3",
            "infrastructure as code terraform language:HCL",
            "terraform provider aws language:HCL",
            "terraform provider azurerm language:HCL"
        ]
        
        # Add service-specific queries for top AWS services
        priority_services = ["ec2", "s3", "rds", "lambda", "vpc", "iam", "eks", "ecs"]
        for service in priority_services:
            base_queries.append(f"terraform aws {service} language:HCL stars:>2")
        
        # Add Azure-specific queries
        azure_services = ["virtual_machine", "storage_account", "app_service", "sql_database"]
        for service in azure_services:
            base_queries.append(f"terraform azurerm {service} language:HCL stars:>2")
        
        return base_queries
    
    @retry_on_failure(max_retries=3, delay=1.0)
    def clone_repository(self, repo_data: Dict, target_dir: str) -> bool:
        """Clone repository with error handling"""
        try:
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
    
    def analyze_repository_structure(self, repo_path: str) -> Dict:
        """Analyze repository structure and Terraform content"""
        analysis = {
            "tf_files": [],
            "tfvars_files": [],
            "directories": [],
            "total_tf_files": 0,
            "total_tfvars_files": 0,
            "aws_resources": set(),
            "azure_resources": set(),
            "total_resources": 0,
            "has_modules": False,
            "has_examples": False,
            "complexity_score": 0
        }
        
        try:
            repo_path = Path(repo_path)
            
            # Find all Terraform files
            tf_files = list(repo_path.rglob("*.tf"))
            tfvars_files = list(repo_path.rglob("*.tfvars"))
            
            analysis["tf_files"] = [str(f.relative_to(repo_path)) for f in tf_files]
            analysis["tfvars_files"] = [str(f.relative_to(repo_path)) for f in tfvars_files]
            analysis["total_tf_files"] = len(tf_files)
            analysis["total_tfvars_files"] = len(tfvars_files)
            
            # Analyze content
            for tf_file in tf_files:
                try:
                    content = tf_file.read_text(encoding='utf-8', errors='ignore')
                    aws_resources, azure_resources = extract_resources_from_terraform(content)
                    
                    analysis["aws_resources"].update(aws_resources)
                    analysis["azure_resources"].update(azure_resources)
                    analysis["total_resources"] += len(aws_resources) + len(azure_resources)
                    
                    # Check for advanced constructs
                    if "module \"" in content:
                        analysis["has_modules"] = True
                    if any(keyword in content.lower() for keyword in ["example", "demo", "tutorial"]):
                        analysis["has_examples"] = True
                    
                    # Calculate complexity score
                    complexity_indicators = [
                        ("dynamic", 2), ("count", 2), ("for_each", 3), 
                        ("locals", 1), ("data", 1), ("output", 1),
                        ("variable", 1), ("module", 3)
                    ]
                    
                    for indicator, weight in complexity_indicators:
                        count = content.lower().count(indicator)
                        analysis["complexity_score"] += count * weight
                        
                except Exception as e:
                    logger.warning(f"Error analyzing file {tf_file}: {e}")
                    continue
            
            # Convert sets to lists for JSON serialization
            analysis["aws_resources"] = list(analysis["aws_resources"])
            analysis["azure_resources"] = list(analysis["azure_resources"])
            
            # Find Terraform directories
            tf_dirs = set()
            for tf_file in tf_files:
                tf_dirs.add(str(tf_file.parent.relative_to(repo_path)))
            analysis["directories"] = list(tf_dirs)
            
        except Exception as e:
            logger.error(f"Error analyzing repository structure: {e}")
        
        return analysis
    
    def filter_quality_repositories(self, repos: List[Dict]) -> List[Dict]:
        """Filter repositories based on quality criteria"""
        quality_repos = []
        
        for repo in repos:
            # Skip repositories that are too small or too large
            if repo.get("size", 0) < 10 or repo.get("size", 0) > 500000:
                continue
            
            # Skip repositories with very few stars (likely low quality)
            if repo.get("stargazers_count", 0) < 1:
                continue
            
            # Skip archived repositories
            if repo.get("archived", False):
                continue
            
            # Prefer repositories with recent activity
            if repo.get("updated_at"):
                try:
                    from datetime import datetime, timedelta
                    updated = datetime.fromisoformat(repo["updated_at"].replace('Z', '+00:00'))
                    if updated < datetime.now().replace(tzinfo=updated.tzinfo) - timedelta(days=365):
                        continue
                except Exception:
                    pass
            
            quality_repos.append(repo)
        
        return quality_repos
    
    def collect_repositories_parallel(self) -> List[Dict]:
        """Collect repositories using parallel processing"""
        logger.info("Starting repository collection process")
        
        queries = self.get_comprehensive_search_queries()
        all_repos = []
        
        # Collect from all queries
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(queries))) as executor:
            future_to_query = {
                executor.submit(self.search_repositories, query): query 
                for query in queries
            }
            
            for future in concurrent.futures.as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    repos = future.result()
                    all_repos.extend(repos)
                    logger.info(f"Query '{query}' returned {len(repos)} repositories")
                except Exception as e:
                    logger.error(f"Query '{query}' failed: {e}")
        
        # Remove duplicates by ID
        unique_repos = {}
        for repo in all_repos:
            repo_id = repo.get("id")
            if repo_id and repo_id not in unique_repos:
                unique_repos[repo_id] = repo
        
        repos_list = list(unique_repos.values())
        logger.info(f"Found {len(repos_list)} unique repositories")
        
        # Filter for quality
        quality_repos = self.filter_quality_repositories(repos_list)
        logger.info(f"Filtered to {len(quality_repos)} quality repositories")
        
        return quality_repos
    
    def analyze_repositories_parallel(self, repos: List[Dict]) -> List[Dict]:
        """Analyze repositories in parallel"""
        logger.info(f"Analyzing {len(repos)} repositories")
        
        analyzed_repos = []
        progress = ProgressTracker(len(repos), "Repository Analysis")
        
        # Limit the number of repositories to analyze based on config
        repos_to_analyze = repos[:config.max_repos]
        
        def analyze_single_repo(repo_data):
            """Analyze a single repository"""
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    repo_dir = os.path.join(temp_dir, "repo")
                    
                    if self.clone_repository(repo_data, repo_dir):
                        analysis = self.analyze_repository_structure(repo_dir)
                        
                        # Only keep repositories with actual Terraform resources
                        if analysis["total_resources"] > 0:
                            repo_analysis = {
                                **repo_data,
                                "analysis": analysis,
                                "collection_timestamp": datetime.now().isoformat(),
                                "quality_score": self.calculate_quality_score(repo_data, analysis)
                            }
                            return repo_analysis
                
                return None
                
            except Exception as e:
                logger.warning(f"Failed to analyze repository {repo_data.get('full_name', 'unknown')}: {e}")
                return None
        
        # Process repositories in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            future_to_repo = {
                executor.submit(analyze_single_repo, repo): repo 
                for repo in repos_to_analyze
            }
            
            for future in concurrent.futures.as_completed(future_to_repo):
                try:
                    result = future.result()
                    if result:
                        analyzed_repos.append(result)
                    progress.update()
                except Exception as e:
                    logger.error(f"Repository analysis failed: {e}")
                    progress.update()
        
        progress.finish()
        
        # Sort by quality score and resource coverage
        analyzed_repos.sort(
            key=lambda x: (
                x["quality_score"],
                len(set(x["analysis"]["aws_resources"]) & set(config.aws_services)),
                len(set(x["analysis"]["azure_resources"]) & set(config.azure_services)),
                x["analysis"]["total_resources"]
            ),
            reverse=True
        )
        
        return analyzed_repos
    
    def calculate_quality_score(self, repo_data: Dict, analysis: Dict) -> float:
        """Calculate repository quality score"""
        score = 0.0
        
        # Star rating (0-40 points)
        stars = repo_data.get("stargazers_count", 0)
        score += min(40, stars * 2)  # 2 points per star, max 40
        
        # Resource diversity (0-30 points)
        aws_coverage = len(set(analysis["aws_resources"]) & set(config.aws_services))
        azure_coverage = len(set(analysis["azure_resources"]) & set(config.azure_services))
        total_coverage = aws_coverage + azure_coverage
        score += min(30, total_coverage * 2)
        
        # Complexity (0-20 points)
        complexity = analysis.get("complexity_score", 0)
        score += min(20, complexity / 5)  # Normalize complexity
        
        # File structure (0-10 points)
        if analysis.get("has_modules"):
            score += 5
        if analysis.get("has_examples"):
            score += 3
        if analysis["total_tfvars_files"] > 0:
            score += 2
        
        return score
    
    def save_repository_metadata(self, repos: List[Dict]):
        """Save repository metadata to file"""
        output_file = self.output_dir / "terraform_repositories.json"
        
        metadata = {
            "collection_date": datetime.now().isoformat(),
            "total_repositories": len(repos),
            "collection_config": {
                "max_repos": config.max_repos,
                "github_token_used": bool(config.github_token)
            },
            "repositories": repos
        }
        
        safe_file_operation("write", str(output_file), metadata)
        logger.info(f"Saved {len(repos)} repository records to {output_file}")
        
        # Also save a summary
        summary_file = self.output_dir / "collection_summary.json"
        summary = self.generate_collection_summary(repos)
        safe_file_operation("write", str(summary_file), summary)
        
        return output_file
    
    def generate_collection_summary(self, repos: List[Dict]) -> Dict:
        """Generate collection summary statistics"""
        aws_services_found = set()
        azure_services_found = set()
        total_resources = 0
        total_stars = 0
        
        complexity_distribution = {"low": 0, "medium": 0, "high": 0}
        size_distribution = {"small": 0, "medium": 0, "large": 0}
        
        for repo in repos:
            analysis = repo.get("analysis", {})
            
            aws_services_found.update(analysis.get("aws_resources", []))
            azure_services_found.update(analysis.get("azure_resources", []))
            total_resources += analysis.get("total_resources", 0)
            total_stars += repo.get("stargazers_count", 0)
            
            # Complexity distribution
            complexity = analysis.get("complexity_score", 0)
            if complexity < 20:
                complexity_distribution["low"] += 1
            elif complexity < 50:
                complexity_distribution["medium"] += 1
            else:
                complexity_distribution["high"] += 1
            
            # Size distribution
            size = repo.get("size", 0)
            if size < 1000:
                size_distribution["small"] += 1
            elif size < 10000:
                size_distribution["medium"] += 1
            else:
                size_distribution["large"] += 1
        
        return {
            "total_repositories": len(repos),
            "total_resources": total_resources,
            "total_stars": total_stars,
            "aws_services_coverage": {
                "found": len(aws_services_found),
                "target": len(config.aws_services),
                "percentage": (len(aws_services_found) / len(config.aws_services)) * 100,
                "services": list(aws_services_found)
            },
            "azure_services_coverage": {
                "found": len(azure_services_found),
                "target": len(config.azure_services),
                "percentage": (len(azure_services_found) / len(config.azure_services)) * 100,
                "services": list(azure_services_found)
            },
            "complexity_distribution": complexity_distribution,
            "size_distribution": size_distribution,
            "average_quality_score": sum(r.get("quality_score", 0) for r in repos) / len(repos) if repos else 0
        }
    
    def run_collection(self) -> str:
        """Main method to run repository collection"""
        logger.info("Starting GitHub repository collection")
        
        try:
            # Search for repositories
            repos = self.collect_repositories_parallel()
            
            # Analyze repositories
            analyzed_repos = self.analyze_repositories_parallel(repos)
            
            # Save results
            output_file = self.save_repository_metadata(analyzed_repos)
            
            logger.info(f"Collection complete! Found {len(analyzed_repos)} quality repositories")
            
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Repository collection failed: {e}")
            raise

def main():
    """Main execution function"""
    import hashlib
    
    collector = GitHubTerraformCollector()
    
    try:
        output_file = collector.run_collection()
        print(f"Repository collection completed successfully!")
        print(f"Results saved to: {output_file}")
        
    except Exception as e:
        logger.error(f"Collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
