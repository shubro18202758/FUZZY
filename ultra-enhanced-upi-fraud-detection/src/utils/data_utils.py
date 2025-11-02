"""
Data utilities for the Ultra Enhanced UPI Fraud Detection System
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

logger = logging.getLogger(__name__)

class DataProcessor:
    """Data processing utilities for your UPI dataset"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
    
    def clean_and_validate(self, df):
        """Clean and validate your dataset"""
        df_clean = df.copy()
        
        # Remove any duplicate transactions
        initial_shape = df_clean.shape
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_shape[0] - df_clean.shape[0]
        
        if duplicates_removed > 0:
            print(f"🧹 Removed {duplicates_removed} duplicate transactions")
        
        # Handle missing values
        missing_counts = df_clean.isnull().sum()
        if missing_counts.sum() > 0:
            print("🔍 Missing values found:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"   {col}: {count} missing values")
                
                # Fill missing values based on column type
                if col in ['trans_amount', 'age']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                elif col in ['category', 'state']:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].mode().iloc[0])
                else:
                    df_clean[col] = df_clean[col].fillna(0)
        
        # Validate data ranges
        self._validate_data_ranges(df_clean)
        
        return df_clean
    
    def _validate_data_ranges(self, df):
        """Validate data ranges for your dataset"""
        validations = [
            ('trans_hour', 0, 23),
            ('trans_day', 1, 31),
            ('trans_month', 1, 12),
            ('trans_year', 2020, 2025),
            ('age', 15, 100),
            ('trans_amount', 0, np.inf)
        ]
        
        for col, min_val, max_val in validations:
            if col in df.columns:
                invalid_mask = (df[col] < min_val) | (df[col] > max_val)
                invalid_count = invalid_mask.sum()
                
                if invalid_count > 0:
                    print(f"⚠️ Found {invalid_count} invalid values in {col}")
                    # Cap outliers
                    df.loc[df[col] < min_val, col] = min_val
                    df.loc[df[col] > max_val, col] = max_val
