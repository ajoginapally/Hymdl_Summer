"""
Example script demonstrating how to use the Terraform prediction pipeline
"""

import json
import time
import logging
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from config.config import config
from config.utils import setup_logging
from scripts.pipeline import TerraformPredictionPipeline

def run_small_example():
    """Run a small example with limited data for demonstration"""
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Terraform prediction example")
    
    # Initialize pipeline
    pipeline = TerraformPredictionPipeline()
    
    try:
        # Run pipeline with small dataset
        results = pipeline.run_full_pipeline(
            max_repos=10,  # Small number for quick demo
            max_iterations=2,
            performance_threshold=0.7  # Lower threshold for demo
        )
        
        # Print results
        print("\n" + "="*60)
        print("EXAMPLE PIPELINE RESULTS")
        print("="*60)
        
        print(f"Status: {results.get('status', 'unknown')}")
        print(f"Duration: {results.get('duration', 0):.1f} seconds")
        print(f"Iterations: {results.get('iterations', 0)}")
        print(f"Final Performance: {results.get('final_performance', 0):.3f}")
        
        if results.get('status') == 'success':
            print("\n✅ Example completed successfully!")
            print("\nNext steps:")
            print("1. Start the API server: python server/api.py")
            print("2. Test predictions: curl http://localhost:8000/health")
            print("3. Run full pipeline with more data: python scripts/pipeline.py --max-repos 100")
        else:
            print(f"\n❌ Example failed: {results.get('error', 'Unknown error')}")
            
    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Example failed: {e}")

def demonstrate_api_usage():
    """Demonstrate API usage with example Terraform code"""
    
    import requests
    import time
    
    # Example Terraform configurations
    examples = [
        {
            "name": "Simple S3 Bucket",
            "code": """
resource "aws_s3_bucket" "example" {
  bucket = "my-example-bucket"
  
  tags = {
    Name        = "Example bucket"
    Environment = "Dev"
  }
}
"""
        },
        {
            "name": "Azure App Service",
            "code": """
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
}
"""
        },
        {
            "name": "AWS VPC with Subnets",
            "code": """
resource "aws_vpc" "main" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  
  tags = {
    Name = "main-vpc"
  }
}

resource "aws_subnet" "public" {
  count             = 2
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.${count.index + 1}.0/24"
  availability_zone = data.aws_availability_zones.available.names[count.index]
  
  map_public_ip_on_launch = true
  
  tags = {
    Name = "public-subnet-${count.index + 1}"
  }
}
"""
        }
    ]
    
    print("\n" + "="*60)
    print("API USAGE DEMONSTRATION")
    print("="*60)
    
    # Check if API is running
    api_url = "http://localhost:8000"
    
    try:
        # Health check
        health_response = requests.get(f"{api_url}/health", timeout=5)
        health_data = health_response.json()
        
        print(f"API Status: {health_data.get('status')}")
        print(f"Model Loaded: {health_data.get('model_loaded')}")
        
        if not health_data.get('model_loaded'):
            print("⚠️  Model not loaded. Make sure to train the model first.")
            return
        
        # Test predictions on examples
        for i, example in enumerate(examples, 1):
            print(f"\n--- Example {i}: {example['name']} ---")
            
            try:
                response = requests.post(
                    f"{api_url}/predict",
                    json={
                        "terraform_code": example['code'],
                        "max_tokens": 512,
                        "temperature": 0.1
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ Prediction successful")
                    print(f"   Confidence: {data['confidence']:.3f}")
                    print(f"   Processing Time: {data['processing_time']:.3f}s")
                    print(f"   Predicted Resources: {len(data['prediction'])}")
                    
                    # Print first prediction for reference
                    if data['prediction']:
                        first = data['prediction'][0]
                        print(f"   First Resource: {first.get('type', 'unknown')}")
                else:
                    print(f"❌ Prediction failed: {response.status_code}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {e}")
            
            time.sleep(1)  # Brief pause between requests
    
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server.")
        print("Make sure the server is running: python server/api.py")

def main():
    """Main function for running examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terraform Prediction Examples")
    parser.add_argument("--mode", choices=["pipeline", "api"], default="pipeline",
                       help="Run pipeline example or API demonstration")
    
    args = parser.parse_args()
    
    if args.mode == "pipeline":
        run_small_example()
    elif args.mode == "api":
        demonstrate_api_usage()

if __name__ == "__main__":
    main()
