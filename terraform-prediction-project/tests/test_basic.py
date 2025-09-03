"""
Basic unit tests for Terraform prediction project components
"""

import unittest
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class TestBasicComponents(unittest.TestCase):
    """Test basic component imports and initialization"""
    
    def test_config_import(self):
        """Test config module import"""
        try:
            from config.config import config
            self.assertIsNotNone(config)
            self.assertTrue(hasattr(config, 'model_name'))
        except ImportError as e:
            self.fail(f"Could not import config: {e}")
    
    def test_utils_import(self):
        """Test utils module import"""
        try:
            from config.utils import setup_logging, safe_file_operation
            self.assertIsNotNone(setup_logging)
            self.assertIsNotNone(safe_file_operation)
        except ImportError as e:
            self.fail(f"Could not import utils: {e}")
    
    def test_github_collector_import(self):
        """Test GitHub collector import"""
        try:
            from scripts.data_collection.github_collector import GitHubTerraformCollector
            collector = GitHubTerraformCollector()
            self.assertIsNotNone(collector)
        except ImportError as e:
            self.fail(f"Could not import GitHub collector: {e}")
    
    def test_terraform_analyzer_import(self):
        """Test Terraform analyzer import"""
        try:
            from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
            analyzer = TerraformAnalyzer()
            self.assertIsNotNone(analyzer)
        except ImportError as e:
            self.fail(f"Could not import Terraform analyzer: {e}")
    
    def test_terraform_analyzer_basic_functionality(self):
        """Test basic Terraform analysis functionality"""
        try:
            from scripts.data_collection.terraform_analyzer import TerraformAnalyzer
            
            analyzer = TerraformAnalyzer()
            
            # Test simple Terraform configuration
            terraform_content = """
resource "aws_s3_bucket" "example" {
  bucket = "my-test-bucket"
  
  tags = {
    Environment = "test"
  }
}
"""
            
            result = analyzer.analyze_terraform_content(terraform_content)
            
            self.assertIsNotNone(result)
            self.assertIn('resources', result)
            self.assertGreater(len(result['resources']), 0)
            
            # Check that S3 bucket was detected
            s3_buckets = [r for r in result['resources'] if r['type'] == 'aws_s3_bucket']
            self.assertEqual(len(s3_buckets), 1)
            self.assertEqual(s3_buckets[0]['name'], 'example')
            
        except Exception as e:
            self.fail(f"Terraform analyzer functionality test failed: {e}")
    
    def test_dataset_processor_import(self):
        """Test dataset processor import"""
        try:
            from scripts.model_training.dataset_processor import TerraformDatasetProcessor
            # Don't initialize here as it requires model download
            self.assertTrue(True)  # Import successful
        except ImportError as e:
            self.fail(f"Could not import dataset processor: {e}")

class TestEnvironmentSetup(unittest.TestCase):
    """Test environment setup and requirements"""
    
    def test_python_version(self):
        """Test Python version is 3.11+"""
        version = sys.version_info
        self.assertGreaterEqual(version.major, 3)
        if version.major == 3:
            self.assertGreaterEqual(version.minor, 11)
    
    def test_required_packages(self):
        """Test that required packages are available"""
        required_packages = [
            'torch',
            'transformers', 
            'datasets',
            'fastapi',
            'jsonlines',
            'pydantic',
            'requests',
            'numpy',
            'sklearn'
        ]
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                self.fail(f"Required package '{package}' not available")
    
    def test_directory_structure(self):
        """Test that required directories exist"""
        required_dirs = [
            'config',
            'scripts',
            'scripts/data_collection',
            'scripts/model_training', 
            'scripts/validation',
            'server',
            'data',
            'models',
            'logs'
        ]
        
        for dir_name in required_dirs:
            dir_path = project_root / dir_name
            self.assertTrue(dir_path.exists(), f"Directory '{dir_name}' does not exist")
            self.assertTrue(dir_path.is_dir(), f"'{dir_name}' is not a directory")

class TestConfiguration(unittest.TestCase):
    """Test configuration settings"""
    
    def test_config_values(self):
        """Test that config has expected values"""
        try:
            from config.config import config
            
            # Test that key configuration values exist
            self.assertIsNotNone(config.model_name)
            self.assertIsNotNone(config.data_dir)
            self.assertIsNotNone(config.model_dir)
            self.assertGreater(config.max_seq_length, 0)
            self.assertGreater(config.num_epochs, 0)
            
        except Exception as e:
            self.fail(f"Configuration test failed: {e}")

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
