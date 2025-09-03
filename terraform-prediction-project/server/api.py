"""
FastAPI server for serving the Terraform prediction model
"""

import json
import logging
import time
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.config import config
from scripts.validation.model_validator import TerraformModelValidator

logger = logging.getLogger(__name__)

# Pydantic models for API requests/responses
class TerraformPredictionRequest(BaseModel):
    terraform_code: str = Field(..., description="Terraform configuration code")
    max_tokens: Optional[int] = Field(512, description="Maximum tokens for prediction")
    temperature: Optional[float] = Field(0.1, description="Sampling temperature")

class TerraformPredictionResponse(BaseModel):
    prediction: List[Dict[str, Any]] = Field(..., description="Predicted resource changes")
    confidence: float = Field(..., description="Model confidence score")
    processing_time: float = Field(..., description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(..., description="Additional metadata")

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
    uptime: float

class ValidationRequest(BaseModel):
    terraform_code: str = Field(..., description="Terraform configuration to validate")
    expected_output: Optional[List[Dict[str, Any]]] = Field(None, description="Expected output for comparison")

class ValidationResponse(BaseModel):
    prediction: List[Dict[str, Any]]
    ground_truth: Optional[List[Dict[str, Any]]]
    metrics: Optional[Dict[str, float]]
    processing_time: float

# Global variables
app = FastAPI(
    title="Terraform Prediction API",
    description="API for predicting Terraform resource changes",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
model_validator: Optional[TerraformModelValidator] = None
model_path: Optional[str] = None
start_time = time.time()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global model_validator, model_path
    
    try:
        # Default model path
        model_path = str(config.model_dir / "fine_tuned")
        
        # Initialize validator (which loads the model)
        model_validator = TerraformModelValidator()
        model_validator.load_model(model_path)
        
        logger.info(f"Model loaded successfully from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Don't raise exception to allow server to start even without model

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model_validator else "degraded",
        model_loaded=model_validator is not None,
        version="1.0.0",
        uptime=time.time() - start_time
    )

@app.post("/predict", response_model=TerraformPredictionResponse)
async def predict_terraform_changes(request: TerraformPredictionRequest):
    """Predict Terraform resource changes"""
    if not model_validator:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time_request = time.time()
    
    try:
        # Run prediction
        prediction = model_validator.predict_single(
            terraform_code=request.terraform_code,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        
        processing_time = time.time() - start_time_request
        
        # Calculate confidence based on prediction structure
        confidence = calculate_prediction_confidence(prediction)
        
        return TerraformPredictionResponse(
            prediction=prediction,
            confidence=confidence,
            processing_time=processing_time,
            metadata={
                "model_path": model_path,
                "tokens_used": len(request.terraform_code.split()),
                "timestamp": time.time()
            }
        )
    
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/validate", response_model=ValidationResponse)
async def validate_terraform_prediction(request: ValidationRequest):
    """Validate a Terraform prediction against ground truth"""
    if not model_validator:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time_request = time.time()
    
    try:
        # Generate prediction
        prediction = model_validator.predict_single(request.terraform_code)
        
        ground_truth = None
        metrics = None
        
        # If expected output provided, calculate metrics
        if request.expected_output:
            ground_truth = request.expected_output
            metrics = model_validator.calculate_metrics(prediction, ground_truth)
        
        processing_time = time.time() - start_time_request
        
        return ValidationResponse(
            prediction=prediction,
            ground_truth=ground_truth,
            metrics=metrics,
            processing_time=processing_time
        )
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

@app.post("/reload-model")
async def reload_model(background_tasks: BackgroundTasks, new_model_path: Optional[str] = None):
    """Reload the model (useful after retraining)"""
    global model_validator, model_path
    
    try:
        target_path = new_model_path or str(config.model_dir / "fine_tuned")
        
        # Reload model in background
        def reload_task():
            global model_validator, model_path
            try:
                model_validator = TerraformModelValidator()
                model_validator.load_model(target_path)
                model_path = target_path
                logger.info(f"Model reloaded from {target_path}")
            except Exception as e:
                logger.error(f"Failed to reload model: {e}")
        
        background_tasks.add_task(reload_task)
        
        return {"status": "reloading", "target_path": target_path}
    
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if not model_validator:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_path": model_path,
        "model_type": "Llama-3.2-3B-Instruct with LoRA",
        "loaded": True,
        "load_time": start_time
    }

def calculate_prediction_confidence(prediction: List[Dict[str, Any]]) -> float:
    """Calculate confidence score for a prediction"""
    if not prediction:
        return 0.0
    
    # Simple heuristic based on prediction structure completeness
    total_score = 0.0
    
    for item in prediction:
        item_score = 0.0
        
        # Check required fields
        if item.get("address"):
            item_score += 0.3
        if item.get("type"):
            item_score += 0.3
        if item.get("change", {}).get("actions"):
            item_score += 0.4
        
        total_score += item_score
    
    # Average confidence across all predicted items
    return min(total_score / len(prediction), 1.0)

def create_app() -> FastAPI:
    """Factory function to create the FastAPI app"""
    return app

def run_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    reload: bool = False,
    log_level: str = "info"
):
    """Run the API server"""
    uvicorn.run(
        "server.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Terraform Prediction API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", help="Log level")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )
