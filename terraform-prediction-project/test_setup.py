#!/usr/bin/env python3
"""
Simple test script to verify the Terraform prediction project setup
"""

import sys
import os
import subprocess
from pathlib import Path
import importlib

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_python_version():
    """Test Python version"""
    print("🐍 Testing Python version...")
    version = sys.version_info
    
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.11+")
        return False

def test_terraform_cli():
    """Test Terraform CLI availability"""
    print("🔧 Testing Terraform CLI...")
    
    try:
        result = subprocess.run(['terraform', 'version'], capture_output=True, text=True)
        if result.returncode == 0:
            version_line = result.stdout.strip().split('\n')[0]
            print(f"✅ {version_line} - OK")
            return True
        else:
            print("❌ Terraform CLI not working")
            return False
    except FileNotFoundError:
        print("❌ Terraform CLI not found. Install with: brew install terraform")
        return False

def test_environment_variables():
    """Test environment variables"""
    print("🔐 Testing environment variables...")
    
    required_vars = ['GITHUB_TOKEN']
    optional_vars = ['HF_TOKEN', 'WANDB_API_KEY', 'AZURE_DEVOPS_TOKEN']
    
    all_good = True
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} - Set")
        else:
            print(f"❌ {var} - Missing (required)")
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} - Set")
        else:
            print(f"⚠️  {var} - Missing (optional)")
    
    return all_good

def test_directory_structure():
    """Test directory structure"""
    print("📁 Testing directory structure...")
    
    required_dirs = [
        'config', 'scripts', 'server', 'data', 'models', 'logs', 'cache',
        'scripts/data_collection', 'scripts/model_training', 'scripts/validation'
    ]
    
    all_good = True
    
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/ - OK")
        else:
            print(f"❌ {dir_name}/ - Missing")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created {dir_name}/")
    
    return all_good

def test_python_imports():
    """Test critical Python imports"""
    print("📦 Testing Python package imports...")
    
    critical_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'HuggingFace Datasets'),
        ('fastapi', 'FastAPI'),
        ('jsonlines', 'JSONL'),
    ]
    
    all_good = True
    
    for package, name in critical_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - Import failed: {e}")
            all_good = False
    
    return all_good

def test_project_imports():
    """Test project-specific imports"""
    print("🏗️ Testing project imports...")
    
    project_modules = [
        ('config.config', 'Project configuration'),
        ('config.utils', 'Project utilities'),
        ('scripts.data_collection.github_collector', 'GitHub collector'),
        ('scripts.data_collection.terraform_analyzer', 'Terraform analyzer'),
        ('scripts.model_training.dataset_processor', 'Dataset processor'),
    ]
    
    all_good = True
    
    for module, name in project_modules:
        try:
            importlib.import_module(module)
            print(f"✅ {name} - OK")
        except ImportError as e:
            print(f"❌ {name} - Import failed: {e}")
            all_good = False
    
    return all_good

def test_model_access():
    """Test model access and download"""
    print("🤖 Testing model access...")
    
    try:
        from transformers import AutoTokenizer
        
        # Test if we can access the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-3B-Instruct",
            trust_remote_code=True
        )
        print("✅ Llama-3.2-3B-Instruct tokenizer - OK")
        print(f"   Vocab size: {len(tokenizer)}")
        return True
        
    except Exception as e:
        print(f"❌ Model access failed: {e}")
        print("   Make sure HF_TOKEN is set and you have access to the model")
        return False

def run_quick_functionality_test():
    """Run a quick end-to-end functionality test"""
    print("⚡ Running quick functionality test...")
    
    try:
        from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
        
        # Test Terraform analysis
        analyzer = TerraformAnalyzer()
        test_terraform = """
resource "aws_s3_bucket" "test" {
  bucket = "test-bucket"
  
  tags = {
    Environment = "test"
  }
}
"""
        
        result = analyzer.analyze_terraform_content(test_terraform)
        
        if result and result.get('resources'):
            print("✅ Terraform analysis - OK")
            print(f"   Found {len(result['resources'])} resources")
            return True
        else:
            print("❌ Terraform analysis - Failed")
            return False
            
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 TERRAFORM PREDICTION PROJECT - SETUP TEST")
    print("=" * 50)
    
    tests = [
        test_python_version,
        test_terraform_cli,
        test_directory_structure,
        test_environment_variables,
        test_python_imports,
        test_project_imports,
        test_model_access,
        run_quick_functionality_test,
    ]
    
    results = []
    
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
            print()
    
    # Summary
    print("=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! You're ready to use the project.")
        print("\nNext steps:")
        print("1. Run a quick example: python examples/run_example.py --mode pipeline")
        print("2. Or start with data collection: make data-collect")
        print("3. Check the SETUP_GUIDE.md for detailed instructions")
    else:
        print(f"⚠️  {passed}/{total} tests passed. Please fix the failing tests before proceeding.")
        print("\nTo fix issues:")
        print("1. Check the error messages above")
        print("2. Refer to SETUP_GUIDE.md for troubleshooting")
        print("3. Make sure all environment variables are set")
    
    print("=" * 50)
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
