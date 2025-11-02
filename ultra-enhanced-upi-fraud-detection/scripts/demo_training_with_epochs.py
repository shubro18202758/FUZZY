"""
Demo script to show ML training with visible epochs
This will demonstrate what actually happens during model training
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

def demo_neural_network_training():
    """Demo function to show neural network training with visible epochs"""
    print("🚀 DEMO: Neural Network Training with Visible Epochs")
    print("📊 This shows what happens during actual ML model training")
    print("=" * 60)
    
    # Load the data
    print("📁 Loading UPI fraud dataset...")
    df = pd.read_csv('data/upi_fraud_dataset.csv')
    print(f"✅ Loaded {len(df)} transactions")
    
    # Prepare features (simplified version)
    feature_columns = ['trans_hour', 'trans_amount', 'category', 'age', 'state', 'zip']
    X = df[feature_columns]
    y = df['fraud_risk']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n📊 Training Data Shape: {X_train_scaled.shape}")
    print(f"📊 Test Data Shape: {X_test_scaled.shape}")
    
    # Create a simple neural network
    print("\n🧠 Building Neural Network Architecture...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(32, activation='relu'),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Model architecture created!")
    model.summary()
    
    # Setup callbacks
    early_stopping = EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )
    
    print("\n🎯 Starting Training - Watch the Epochs!")
    print("=" * 50)
    print("📈 You'll see: Epoch X/Y - loss: X.XXXX - accuracy: X.XXXX - val_loss: X.XXXX - val_accuracy: X.XXXX")
    print("⏱️  Each epoch processes the entire training dataset once")
    print()
    
    # Train the model with visible epochs
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_test_scaled, y_test),
        epochs=20,  # Reduced for demo
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1  # This shows the epoch progress!
    )
    
    # Final evaluation
    print("\n📊 Final Model Performance:")
    test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"🎯 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"📉 Test Loss: {test_loss:.4f}")
    
    print("\n✨ Training Complete!")
    print("🔍 This is what happens behind the scenes when you train ML models!")
    
    return model, history

if __name__ == "__main__":
    demo_neural_network_training()
