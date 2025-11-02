"""
Ultra Advanced Model Configuration for UPI Fraud Detection
Optimized for your dataset structure
"""
import os
from typing import Dict, List, Any

class UltraModelConfig:
    """Ultra enhanced configuration for your UPI dataset"""
    
    # Dataset Configuration (Your actual dataset structure)
    DATASET_PATH = 'data/upi_fraud_dataset.csv'
    TARGET_COLUMN = 'fraud_risk'
    ID_COLUMNS = ['Id', 'upi_number']
    
    # Your dataset columns
    TEMPORAL_FEATURES = ['trans_hour', 'trans_day', 'trans_month', 'trans_year']
    CATEGORICAL_FEATURES = ['category', 'state']
    NUMERICAL_FEATURES = ['age', 'trans_amount', 'zip']
    
    # Model Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    VALIDATION_SIZE = 0.2
    
    # Advanced Feature Engineering
    TEMPORAL_WINDOWS = [1, 3, 7, 14, 30, 90]  # days
    RISK_THRESHOLDS = {'low': 0.3, 'medium': 0.7, 'high': 0.9}
    
    # Ultra Advanced Ensemble Parameters
    LGBM_PARAMS = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'class_weight': 'balanced',
        'n_estimators': 300
    }
    
    XGB_PARAMS = {
        'n_estimators': 300,
        'max_depth': 8,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss'
    }
    
    RF_PARAMS = {
        'n_estimators': 300,
        'max_depth': 15,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'class_weight': 'balanced',
        'n_jobs': -1
    }
    
    # Real-time Processing
    ALERT_THRESHOLD = 0.7
    CACHE_TTL = 3600
    BATCH_PROCESSING_SIZE = 100
    
    # Monitoring Parameters
    DRIFT_THRESHOLD = 0.05
    PERFORMANCE_THRESHOLD = 0.1
    MONITORING_FREQUENCY = 24  # hours

class DeploymentConfig:
    """Deployment configuration"""
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))
    
    # Redis Configuration
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    
    # Kafka Configuration
    KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
    KAFKA_TOPIC_TRANSACTIONS = 'upi_transactions'
    KAFKA_TOPIC_FRAUD_ALERTS = 'fraud_alerts'
