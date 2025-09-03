"""
Main pipeline orchestrator that coordinates all components
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import config
from config.utils import setup_logging, safe_file_operation
from scripts.data_collection.ground_truth_generator import GroundTruthGenerator
from scripts.model_training.dataset_processor import TerraformDatasetProcessor
from scripts.model_training.model_setup import TerraformModelSetup
from scripts.model_training.trainer import TerraformTrainer
from scripts.validation.model_validator import TerraformModelValidator
from scripts.validation.error_mitigation import ErrorMitigationLoop

logger = logging.getLogger(__name__)

class TerraformPredictionPipeline:
    """Complete pipeline for Terraform prediction model"""
    
    def __init__(self):
        self.ground_truth_generator = GroundTruthGenerator()
        self.dataset_processor = TerraformDatasetProcessor()
        self.model_setup = TerraformModelSetup()
        self.trainer = TerraformTrainer()
        self.validator = TerraformModelValidator()
        self.error_mitigator = ErrorMitigationLoop()
        
    def run_data_collection(self, max_repos: int = 100) -> Dict[str, Any]:
        """Run data collection phase"""
        logger.info(f"Starting data collection for {max_repos} repositories")
        
        try:
            # Generate ground truth dataset
            output_file = config.data_dir / "ground_truth" / "terraform_dataset.json.gz"
            
            self.ground_truth_generator.generate_dataset(
                output_file=str(output_file),
                max_repos=max_repos
            )
            
            # Verify dataset was created
            if not output_file.exists():
                raise FileNotFoundError(f"Dataset file not created: {output_file}")
                
            logger.info(f"Data collection completed. Dataset saved to {output_file}")
            
            # Return dataset statistics
            dataset_stats = self._get_dataset_stats(str(output_file))
            return {"status": "success", "stats": dataset_stats}
            
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_data_processing(self, dataset_file: Optional[str] = None) -> Dict[str, Any]:
        """Run dataset processing phase"""
        logger.info("Starting dataset processing")
        
        try:
            if not dataset_file:
                dataset_file = str(config.data_dir / "ground_truth" / "terraform_dataset.json.gz")
            
            # Process dataset
            results = self.dataset_processor.process_complete_dataset(dataset_file)
            
            logger.info("Dataset processing completed")
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Dataset processing failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_model_training(self, resume_from_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Run model training phase"""
        logger.info("Starting model training")
        
        try:
            # Setup model and tokenizer
            model, tokenizer = self.model_setup.setup_model_and_tokenizer()
            
            # Train model
            results = self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            logger.info("Model training completed")
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_model_validation(self, model_path: Optional[str] = None) -> Dict[str, Any]:
        """Run model validation phase"""
        logger.info("Starting model validation")
        
        try:
            if not model_path:
                model_path = str(config.model_dir / "fine_tuned")
            
            # Run validation
            results = self.validator.validate_model(
                model_path=model_path,
                test_dataset_path=str(config.data_dir / "processed" / "test.jsonl")
            )
            
            logger.info("Model validation completed")
            return {"status": "success", "results": results}
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_error_mitigation(self, validation_results_file: str) -> Dict[str, Any]:
        """Run error mitigation cycle"""
        logger.info("Starting error mitigation")
        
        try:
            # Analyze errors and generate augmentation samples
            improvements_made = self.error_mitigator.run_mitigation_cycle(
                validation_results_file
            )
            
            status = "improved" if improvements_made else "no_improvements"
            logger.info(f"Error mitigation completed: {status}")
            
            return {"status": "success", "improvements_made": improvements_made}
            
        except Exception as e:
            logger.error(f"Error mitigation failed: {e}")
            return {"status": "failed", "error": str(e)}
    
    def run_full_pipeline(
        self, 
        max_repos: int = 100,
        max_iterations: int = 3,
        performance_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """Run the complete pipeline with iterative improvement"""
        logger.info("Starting full Terraform prediction pipeline")
        
        pipeline_results = {
            "start_time": time.time(),
            "phases": {},
            "final_performance": None,
            "iterations": 0
        }
        
        try:
            # Phase 1: Data Collection
            logger.info("="*50)
            logger.info("PHASE 1: Data Collection")
            logger.info("="*50)
            
            collection_results = self.run_data_collection(max_repos)
            pipeline_results["phases"]["data_collection"] = collection_results
            
            if collection_results["status"] != "success":
                return pipeline_results
            
            # Phase 2: Data Processing
            logger.info("="*50)
            logger.info("PHASE 2: Data Processing")
            logger.info("="*50)
            
            processing_results = self.run_data_processing()
            pipeline_results["phases"]["data_processing"] = processing_results
            
            if processing_results["status"] != "success":
                return pipeline_results
            
            # Iterative training and validation with error mitigation
            iteration = 0
            best_performance = 0.0
            
            while iteration < max_iterations:
                iteration += 1
                pipeline_results["iterations"] = iteration
                
                logger.info("="*50)
                logger.info(f"ITERATION {iteration}: Training & Validation")
                logger.info("="*50)
                
                # Phase 3: Model Training
                training_results = self.run_model_training()
                pipeline_results["phases"][f"training_iter_{iteration}"] = training_results
                
                if training_results["status"] != "success":
                    break
                
                # Phase 4: Model Validation
                validation_results = self.run_model_validation()
                pipeline_results["phases"][f"validation_iter_{iteration}"] = validation_results
                
                if validation_results["status"] != "success":
                    break
                
                # Check performance
                current_performance = validation_results["results"].get("overall_f1", 0.0)
                pipeline_results["final_performance"] = current_performance
                
                logger.info(f"Iteration {iteration} performance: {current_performance:.3f}")
                
                # Check if we've reached the performance threshold
                if current_performance >= performance_threshold:
                    logger.info(f"Performance threshold {performance_threshold} reached!")
                    break
                
                # Check if performance improved significantly
                if current_performance > best_performance + 0.05:
                    best_performance = current_performance
                    logger.info("Performance improved, continuing iteration")
                else:
                    logger.info("No significant improvement, stopping iterations")
                    break
                
                # Phase 5: Error Mitigation (if not last iteration)
                if iteration < max_iterations:
                    validation_file = str(config.model_dir / "validation_results.json")
                    mitigation_results = self.run_error_mitigation(validation_file)
                    pipeline_results["phases"][f"mitigation_iter_{iteration}"] = mitigation_results
                    
                    if not mitigation_results.get("improvements_made", False):
                        logger.info("No mitigation improvements available")
                        break
            
            # Final summary
            pipeline_results["end_time"] = time.time()
            pipeline_results["duration"] = pipeline_results["end_time"] - pipeline_results["start_time"]
            
            self._save_pipeline_results(pipeline_results)
            self._print_pipeline_summary(pipeline_results)
            
            return pipeline_results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_results["status"] = "failed"
            pipeline_results["error"] = str(e)
            return pipeline_results
    
    def _get_dataset_stats(self, dataset_file: str) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        try:
            import gzip
            
            with gzip.open(dataset_file, 'rt') as f:
                data = json.load(f)
            
            return {
                "total_samples": len(data),
                "repositories": len(set(item.get("repository", "") for item in data)),
                "providers": list(set(item.get("metadata", {}).get("provider", "") for item in data))
            }
        except Exception as e:
            logger.warning(f"Could not get dataset stats: {e}")
            return {"error": str(e)}
    
    def _save_pipeline_results(self, results: Dict[str, Any]) -> None:
        """Save pipeline results to file"""
        results_file = config.model_dir / "pipeline_results.json"
        
        with safe_file_operation(str(results_file), 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to {results_file}")
    
    def _print_pipeline_summary(self, results: Dict[str, Any]) -> None:
        """Print a summary of pipeline results"""
        print("\n" + "="*60)
        print("TERRAFORM PREDICTION PIPELINE SUMMARY")
        print("="*60)
        
        print(f"Duration: {results.get('duration', 0):.1f} seconds")
        print(f"Iterations: {results.get('iterations', 0)}")
        print(f"Final Performance: {results.get('final_performance', 0):.3f}")
        
        print("\nPhase Results:")
        for phase, result in results.get("phases", {}).items():
            status = result.get("status", "unknown")
            print(f"  {phase}: {status}")
        
        print("\n" + "="*60)

def main():
    """Main entry point for the pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Terraform Prediction Model Pipeline")
    parser.add_argument("--phase", choices=[
        "data_collection", "data_processing", "training", 
        "validation", "error_mitigation", "full"
    ], default="full", help="Pipeline phase to run")
    parser.add_argument("--max-repos", type=int, default=100, 
                       help="Maximum repositories for data collection")
    parser.add_argument("--max-iterations", type=int, default=3, 
                       help="Maximum training iterations")
    parser.add_argument("--performance-threshold", type=float, default=0.8, 
                       help="Performance threshold to stop iterations")
    parser.add_argument("--resume-from", type=str, 
                       help="Resume training from checkpoint")
    parser.add_argument("--model-path", type=str, 
                       help="Path to model for validation")
    parser.add_argument("--validation-file", type=str, 
                       help="Validation results file for error mitigation")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Initialize pipeline
    pipeline = TerraformPredictionPipeline()
    
    try:
        if args.phase == "data_collection":
            results = pipeline.run_data_collection(args.max_repos)
            
        elif args.phase == "data_processing":
            results = pipeline.run_data_processing()
            
        elif args.phase == "training":
            results = pipeline.run_model_training(args.resume_from)
            
        elif args.phase == "validation":
            results = pipeline.run_model_validation(args.model_path)
            
        elif args.phase == "error_mitigation":
            if not args.validation_file:
                logger.error("--validation-file required for error mitigation")
                return
            results = pipeline.run_error_mitigation(args.validation_file)
            
        elif args.phase == "full":
            results = pipeline.run_full_pipeline(
                max_repos=args.max_repos,
                max_iterations=args.max_iterations,
                performance_threshold=args.performance_threshold
            )
        
        # Print final status
        status = results.get("status", "unknown")
        print(f"\nPipeline phase '{args.phase}' completed with status: {status}")
        
        if status == "failed":
            print(f"Error: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        print("\nPipeline interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {e}")
        print(f"\nPipeline failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
