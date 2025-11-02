"""
Training script for Ultra Advanced UPI Fraud Detection
Run this to train the model with your dataset
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.ultra_advanced_detector import UltraAdvancedUPIFraudDetector
import pandas as pd

def main():
    """Main training function"""
    print("🚀 Ultra Advanced UPI Fraud Detection Training")
    print("🎯 Training with Your Actual Dataset")
    print("=" * 60)
    
    # Check if dataset exists
    dataset_path = 'data/upi_fraud_dataset.csv'
    if not os.path.exists(dataset_path):
        print("❌ Error: 'data/upi_fraud_dataset.csv' not found!")
        print("📁 Please ensure your dataset file is in the data/ folder")
        return
    
    # Initialize detector
    detector = UltraAdvancedUPIFraudDetector()
    
    # Train with your dataset
    detector.fit(dataset_path)
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save the trained model
    detector.save_model('models/ultra_advanced_upi_detector.pkl')
    
    print("\n✅ Training completed successfully!")
    print("💾 Model saved as: models/ultra_advanced_upi_detector.pkl")
    print("🚀 Ready for deployment!")

if __name__ == "__main__":
    main()
