#!/usr/bin/env python3
"""
Basic demonstration of Terraform analysis without requiring trained models
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def demo_terraform_analysis():
    """Demonstrate Terraform configuration analysis"""
    print("🔍 TERRAFORM ANALYSIS DEMO")
    print("=" * 40)
    
    try:
        from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
        
        # Initialize analyzer
        analyzer = TerraformAnalyzer()
        print("✅ Terraform analyzer initialized")
        
        # Example configurations to analyze
        examples = {
            "AWS S3 Bucket": """
resource "aws_s3_bucket" "website" {
  bucket = "my-website-bucket"
  
  tags = {
    Environment = "production"
    Purpose     = "website"
  }
}

resource "aws_s3_bucket_versioning" "website" {
  bucket = aws_s3_bucket.website.id
  versioning_configuration {
    status = "Enabled"
  }
}
""",
            
            "Azure Resource Group": """
resource "azurerm_resource_group" "main" {
  name     = "production-rg"
  location = "East US"
}

resource "azurerm_storage_account" "main" {
  name                     = "prodstorageaccount"
  resource_group_name      = azurerm_resource_group.main.name
  location                 = azurerm_resource_group.main.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
}
""",
            
            "AWS VPC with Dynamic Blocks": """
variable "availability_zones" {
  default = ["us-west-2a", "us-west-2b", "us-west-2c"]
}

resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public" {
  count             = length(var.availability_zones)
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = var.availability_zones[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "public-subnet-${count.index + 1}"
    Type = "public"
  }
}
"""
        }
        
        # Analyze each example
        for name, terraform_code in examples.items():
            print(f"\n--- {name} ---")
            
            try:
                result = analyzer.analyze_terraform_content(terraform_code)
                
                print(f"📊 Analysis Results:")
                print(f"   Resources found: {len(result['resources'])}")
                print(f"   Variables: {len(result['variables'])}")
                print(f"   Complexity score: {result['complexity_score']}")
                print(f"   Primary provider: {result.get('primary_provider', 'unknown')}")
                
                print(f"📦 Resource Details:")
                for resource in result['resources']:
                    print(f"   • {resource['type']}.{resource['name']}")
                
                if result['variables']:
                    print(f"🔧 Variables:")
                    for var in result['variables']:
                        print(f"   • {var['name']} ({var.get('type', 'any')})")
                
                # Check for special constructs
                special_features = []
                if result.get('has_dynamic_blocks'):
                    special_features.append("dynamic blocks")
                if result.get('has_count'):
                    special_features.append("count")
                if result.get('has_for_each'):
                    special_features.append("for_each")
                if result.get('has_dependencies'):
                    special_features.append("resource dependencies")
                
                if special_features:
                    print(f"✨ Special features: {', '.join(special_features)}")
                
            except Exception as e:
                print(f"❌ Analysis failed: {e}")
        
        print("\n✅ Terraform analysis demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_data_collection():
    """Demonstrate data collection capabilities"""
    print("\n🌐 GITHUB DATA COLLECTION DEMO")
    print("=" * 40)
    
    try:
        from scripts.data_collection.github_collector import GitHubTerraformCollector
        
        # Initialize collector
        collector = GitHubTerraformCollector()
        print("✅ GitHub collector initialized")
        
        # Search for repositories (without cloning)
        print("🔍 Searching for Terraform repositories...")
        
        search_query = "terraform aws language:hcl size:<1000"
        repos = collector.search_repositories(search_query, max_results=5)
        
        print(f"📦 Found {len(repos)} repositories:")
        for i, repo in enumerate(repos[:3], 1):  # Show first 3
            print(f"   {i}. {repo['full_name']}")
            print(f"      Stars: {repo['stargazers_count']}")
            print(f"      Size: {repo['size']} KB")
            print(f"      Language: {repo.get('language', 'N/A')}")
            print(f"      URL: {repo['html_url']}")
            print()
        
        print("✅ GitHub data collection demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("   Make sure GITHUB_TOKEN is set correctly")
        return False

def demo_configuration():
    """Demonstrate configuration system"""
    print("\n⚙️ CONFIGURATION DEMO") 
    print("=" * 40)
    
    try:
        from config.config import config
        
        print("📋 Current Configuration:")
        print(f"   Model name: {config.model_name}")
        print(f"   Max sequence length: {config.max_seq_length}")
        print(f"   Data directory: {config.data_dir}")
        print(f"   Model directory: {config.model_dir}")
        print(f"   Training epochs: {config.num_epochs}")
        print(f"   Learning rate: {config.learning_rate}")
        print(f"   Batch size: {config.per_device_train_batch_size}")
        
        print("\n🎯 Performance Settings:")
        print(f"   LoRA rank: {config.lora_r}")
        print(f"   LoRA alpha: {config.lora_alpha}")
        print(f"   Max repos per search: {config.max_repos_per_search}")
        print(f"   Cache TTL: {config.cache_ttl_hours} hours")
        
        print("\n✅ Configuration demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def demo_utilities():
    """Demonstrate utility functions"""
    print("\n🛠️ UTILITIES DEMO")
    print("=" * 40)
    
    try:
        from config.utils import safe_file_operation, ProgressTracker
        import tempfile
        import time
        
        # Test safe file operations
        print("📁 Testing file operations...")
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
            test_data = {"test": "data", "numbers": [1, 2, 3]}
            
            # Test writing
            result = safe_file_operation("write", tmp.name, test_data)
            print(f"   Write operation: {'✅ Success' if result else '❌ Failed'}")
            
            # Test reading
            read_data = safe_file_operation("read", tmp.name)
            print(f"   Read operation: {'✅ Success' if read_data == test_data else '❌ Failed'}")
        
        # Test progress tracker
        print("📊 Testing progress tracker...")
        
        tracker = ProgressTracker(total=10, description="Demo progress")
        for i in range(10):
            time.sleep(0.1)  # Simulate work
            tracker.update()
        tracker.finish()
        
        print("✅ Utilities demo completed!")
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

def main():
    """Run all demos"""
    print("🎪 TERRAFORM PREDICTION PROJECT - BASIC DEMOS")
    print("=" * 50)
    
    demos = [
        ("Configuration", demo_configuration),
        ("Utilities", demo_utilities),
        ("Terraform Analysis", demo_terraform_analysis),
        ("Data Collection", demo_data_collection),
    ]
    
    results = []
    
    for name, demo_func in demos:
        print(f"\n🎯 Running {name} Demo...")
        try:
            result = demo_func()
            results.append(result)
        except Exception as e:
            print(f"❌ {name} demo failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 DEMO SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (name, _) in enumerate(demos):
        status = "✅ PASSED" if results[i] else "❌ FAILED"
        print(f"   {name}: {status}")
    
    print()
    if passed == total:
        print("🎉 ALL DEMOS PASSED!")
        print("\nThe project is working correctly. Try running:")
        print("   python examples/run_example.py --mode pipeline")
        print("   make pipeline-small")
    else:
        print(f"⚠️  {passed}/{total} demos passed.")
        print("Please check the error messages and fix any issues.")
    
    print("=" * 50)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
