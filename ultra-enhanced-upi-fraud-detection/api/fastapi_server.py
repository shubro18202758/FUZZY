"""
FastAPI Server for Ultra Advanced UPI Fraud Detection
Optimized for your dataset structure
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import asyncio
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your ultra advanced detector
from src.core.ultra_advanced_detector import UltraAdvancedUPIFraudDetector
from src.realtime.real_time_monitor import UltraRealTimeMonitor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="REVOLUTIONARY Ultra Advanced UPI Fraud Detection API",
    description="API for real-time UPI fraud detection using REVOLUTIONARY feature engineering",
    version="3.0.0 - REVOLUTIONARY EDITION"
)

# Global variables for models
detector = None
monitor = None

class UPITransaction(BaseModel):
    """UPI Transaction model matching your dataset structure"""
    trans_hour: int = Field(..., ge=0, le=23, description="Transaction hour (0-23)")
    trans_day: int = Field(..., ge=1, le=31, description="Transaction day (1-31)")
    trans_month: int = Field(..., ge=1, le=12, description="Transaction month (1-12)")
    trans_year: int = Field(..., ge=2020, le=2030, description="Transaction year")
    category: int = Field(..., ge=0, description="Transaction category")
    upi_number: str = Field(..., description="UPI number")
    age: int = Field(..., ge=15, le=100, description="User age")
    trans_amount: float = Field(..., gt=0, description="Transaction amount")
    state: int = Field(..., ge=0, description="State code")
    zip: int = Field(..., ge=1000, le=99999, description="ZIP code (matches your dataset range)")
    transaction_id: Optional[str] = Field(None, description="Optional transaction ID")

class FraudPredictionResponse(BaseModel):
    """Response model for fraud prediction"""
    transaction_id: str
    fraud_probability: float
    prediction: int
    risk_level: str
    recommended_action: str
    alert: bool
    processing_time_ms: float
    timestamp: str
    transaction_details: Dict[str, Any]
    model_version: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global detector, monitor
    
    try:
        # Use absolute path for model loading
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'models', 'ultra_advanced_upi_detector.pkl')
        detector = UltraAdvancedUPIFraudDetector.load_model(model_path)
        monitor = UltraRealTimeMonitor(detector, alert_threshold=0.7)
        logger.info("✅ Ultra Advanced UPI Fraud Detection models loaded successfully")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        detector = None
        monitor = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if detector is not None else "unhealthy",
        "model_loaded": detector is not None,
        "api_version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=FraudPredictionResponse)
async def predict_fraud(transaction: UPITransaction):
    """
    Predict fraud for a single UPI transaction
    Uses your exact dataset structure
    """
    if detector is None or monitor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Convert to dictionary
        transaction_data = transaction.dict()
        
        # Process with ultra advanced monitor
        result = await monitor.process_single_transaction(transaction_data)
        
        if 'error' in result:
            raise HTTPException(status_code=500, detail=result['error'])
        
        return FraudPredictionResponse(**result)
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict")
async def batch_predict_fraud(transactions: List[UPITransaction]):
    """
    Predict fraud for multiple UPI transactions
    Efficient batch processing
    """
    if detector is None or monitor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    start_time = datetime.now()
    
    try:
        # Convert transactions to list of dictionaries
        transactions_data = [txn.dict() for txn in transactions]
        
        # Process batch
        results = monitor.batch_process_transactions(transactions_data)
        
        # Calculate summary statistics
        total_processed = len(results)
        fraud_detected = sum(1 for r in results if r.get('prediction') == 1)
        alerts_generated = sum(1 for r in results if r.get('alert', False))
        high_risk_count = sum(1 for r in results if r.get('risk_level') in ['HIGH', 'CRITICAL'])
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        summary = {
            'total_transactions': total_processed,
            'fraud_detected': fraud_detected,
            'fraud_rate': round((fraud_detected / total_processed) * 100, 2) if total_processed > 0 else 0,
            'alerts_generated': alerts_generated,
            'alert_rate': round((alerts_generated / total_processed) * 100, 2) if total_processed > 0 else 0,
            'high_risk_transactions': high_risk_count,
            'avg_processing_time_per_txn': round(processing_time / total_processed, 2) if total_processed > 0 else 0
        }
        
        return {
            'results': [result for result in results if 'error' not in result],
            'summary': summary,
            'total_processed': total_processed,
            'processing_time_ms': round(processing_time, 2)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/statistics")
async def get_processing_statistics():
    """Get processing statistics"""
    if monitor is None:
        raise HTTPException(status_code=503, detail="Monitor not initialized")
    
    return monitor.get_processing_statistics()

@app.get("/model_info")
async def get_model_info():
    """Get model information"""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    feature_importance = None
    if detector.feature_importance is not None:
        feature_importance = detector.feature_importance.head(10).to_dict('records')
    
    return {
        "model_type": "Ultra Advanced Ensemble",
        "version": "2.0.0",
        "features_count": len(detector.feature_names) if detector.feature_names else 0,
        "training_metrics": detector.training_metrics,
        "top_features": feature_importance,
        "deployment_date": datetime.now().isoformat(),
        "status": "active"
    }

# Test endpoint for your specific dataset
@app.post("/test_with_sample")
async def test_with_sample_data():
    """Test endpoint with sample data from your dataset structure"""
    if detector is None or monitor is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    # Sample transaction matching your dataset structure (from your actual data)
    sample_transaction = {
        "trans_hour": 23,
        "trans_day": 15,
        "trans_month": 7,
        "trans_year": 2022,
        "category": 12,
        "upi_number": "9957000001",
        "age": 54,
        "trans_amount": 66.21,
        "state": 22,
        "zip": 49879,
        "transaction_id": "test_sample_001"
    }
    
    # Use sync processing instead of async
    result = monitor.process_transaction_sync(sample_transaction)
    
    return {
        "message": "Test completed successfully with your dataset sample",
        "sample_transaction": sample_transaction,
        "prediction_result": result
    }

if __name__ == "__main__":
    import uvicorn
    
    # Load models before starting server
    print("🚀 Loading Ultra Advanced UPI Fraud Detection Models...")
    try:
        # Use absolute path for model loading
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 'models', 'ultra_advanced_upi_detector.pkl')
        print(f"📁 Looking for model at: {model_path}")
        detector = UltraAdvancedUPIFraudDetector.load_model(model_path)
        monitor = UltraRealTimeMonitor(detector, alert_threshold=0.7)
        print("✅ Models loaded successfully!")
        print("🌐 Starting FastAPI server on port 8001...")
    except Exception as e:
        print(f"❌ Failed to load models: {e}")
        detector = None
        monitor = None
    
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=False)
