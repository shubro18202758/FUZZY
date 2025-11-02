"""
Real-time UPI Transaction Monitoring System
"""
import asyncio
import json
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, Any
import time

logger = logging.getLogger(__name__)

class UltraRealTimeMonitor:
    """
    Ultra Advanced Real-time UPI Transaction Monitor
    Optimized for your dataset structure
    """
    
    def __init__(self, detector, alert_threshold=0.7):
        self.detector = detector
        self.alert_threshold = alert_threshold
        self.processing_stats = {
            'total_processed': 0,
            'fraud_detected': 0,
            'alerts_generated': 0,
            'avg_processing_time': 0.0,
            'high_risk_transactions': 0
        }
        self.transaction_log = []
        
    async def process_single_transaction(self, transaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single UPI transaction in real-time
        
        Expected transaction_data format (matching your dataset):
        {
            'trans_hour': int,
            'trans_day': int, 
            'trans_month': int,
            'trans_year': int,
            'category': int,
            'upi_number': str,
            'age': int,
            'trans_amount': float,
            'state': int,
            'zip': int
        }
        """
        start_time = time.time()
        
        try:
            # Convert to DataFrame (matching your dataset structure)
            transaction_df = pd.DataFrame([transaction_data])
            
            # Get ultra advanced fraud prediction
            fraud_probability = self.detector.predict_ultra_advanced(transaction_df)[0]
            prediction = int(fraud_probability > 0.5)
            
            # Advanced risk assessment
            risk_level, recommended_action = self._assess_risk_level(fraud_probability)
            
            # Create comprehensive result
            result = {
                'transaction_id': transaction_data.get('transaction_id', f"txn_{int(time.time() * 1000)}"),
                'fraud_probability': round(float(fraud_probability), 4),
                'prediction': prediction,
                'risk_level': risk_level,
                'recommended_action': recommended_action,
                'alert': fraud_probability >= self.alert_threshold,
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat(),
                'transaction_details': {
                    'amount': transaction_data.get('trans_amount', 0),
                    'hour': transaction_data.get('trans_hour', 0),
                    'category': transaction_data.get('category', 0),
                    'state': transaction_data.get('state', 0),
                    'age': transaction_data.get('age', 0)
                },
                'model_version': 'ultra_advanced_v2.0'
            }
            
            # Update processing statistics
            self._update_stats(result, time.time() - start_time)
            
            # Log transaction
            self.transaction_log.append(result)
            
            # Keep only last 1000 transactions in memory
            if len(self.transaction_log) > 1000:
                self.transaction_log = self.transaction_log[-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing transaction: {e}")
            return {
                'transaction_id': transaction_data.get('transaction_id', 'unknown'),
                'error': str(e),
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round((time.time() - start_time) * 1000, 2),
                'status': 'error'
            }
    
    def _assess_risk_level(self, fraud_probability: float) -> tuple:
        """Assess risk level and recommend action"""
        if fraud_probability >= 0.95:
            return 'CRITICAL', 'BLOCK_IMMEDIATELY'
        elif fraud_probability >= 0.8:
            return 'HIGH', 'MANUAL_REVIEW'
        elif fraud_probability >= 0.6:
            return 'MEDIUM', 'ENHANCED_MONITORING'
        elif fraud_probability >= 0.3:
            return 'LOW', 'ROUTINE_MONITORING'
        else:
            return 'MINIMAL', 'ALLOW'
    
    def _update_stats(self, result: Dict[str, Any], processing_time: float):
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        
        if result['prediction'] == 1:
            self.processing_stats['fraud_detected'] += 1
        
        if result['alert']:
            self.processing_stats['alerts_generated'] += 1
        
        if result['risk_level'] in ['HIGH', 'CRITICAL']:
            self.processing_stats['high_risk_transactions'] += 1
        
        # Update average processing time
        total = self.processing_stats['total_processed']
        current_avg = self.processing_stats['avg_processing_time']
        self.processing_stats['avg_processing_time'] = (
            (current_avg * (total - 1) + processing_time * 1000) / total
        )
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        total = max(self.processing_stats['total_processed'], 1)
        
        return {
            **self.processing_stats,
            'fraud_detection_rate': round((self.processing_stats['fraud_detected'] / total) * 100, 2),
            'alert_rate': round((self.processing_stats['alerts_generated'] / total) * 100, 2),
            'high_risk_rate': round((self.processing_stats['high_risk_transactions'] / total) * 100, 2),
            'avg_processing_time_ms': round(self.processing_stats['avg_processing_time'], 2)
        }
    
    def batch_process_transactions(self, transactions: list) -> list:
        """Process multiple transactions efficiently (Jupyter-compatible version)"""
        results = []
        
        for transaction in transactions:
            # Use synchronous processing for Jupyter compatibility
            result = self.process_transaction_sync(transaction)
            results.append(result)
        
        return results
    
    def process_transaction_sync(self, transaction: dict) -> dict:
        """Synchronous version of transaction processing for Jupyter compatibility"""
        import time
        start_time = time.time()
        
        try:
            # Create DataFrame from transaction
            transaction_df = pd.DataFrame([transaction])
            
            # Prepare data for prediction
            prepared_data = self.detector.prepare_data(transaction_df, is_training=False)
            
            # Apply feature engineering
            enhanced_data = self.detector.feature_engineer.apply_ultra_feature_engineering(prepared_data)
            
            # Make prediction
            probability = self.detector.predict_proba(enhanced_data)[0][1]
            prediction = int(probability > 0.5)
            
            # Determine risk level
            if probability >= 0.8:
                risk_level = "high"
            elif probability >= 0.5:
                risk_level = "medium"
            else:
                risk_level = "low"
            
            processing_time = (time.time() - start_time) * 1000  # Convert to ms
            
            # Create alert if needed
            alert = None
            if probability >= self.alert_threshold:
                alert = {
                    "timestamp": time.time(),
                    "risk_score": probability,
                    "transaction_details": transaction
                }
                self.alerts.append(alert)
            
            return {
                "transaction_id": transaction.get('transaction_id', 'unknown'),
                "fraud_probability": probability,
                "prediction": prediction,
                "risk_level": risk_level,
                "processing_time_ms": processing_time,
                "alert": alert,
                "transaction_details": {
                    "amount": transaction.get('trans_amount', 0),
                    "category": transaction.get('category', 'unknown'),
                    "state": transaction.get('state', 'unknown')
                }
            }
            
        except Exception as e:
            return {
                "transaction_id": transaction.get('transaction_id', 'unknown'),
                "error": str(e),
                "processing_time_ms": (time.time() - start_time) * 1000
            }
    
    def get_recent_alerts(self, limit: int = 10) -> list:
        """Get recent high-risk transactions"""
        alerts = [log for log in self.transaction_log if log.get('alert', False)]
        return sorted(alerts, key=lambda x: x['timestamp'], reverse=True)[:limit]
