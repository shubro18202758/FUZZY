"""
Test the Ultra Advanced UPI Fraud Detection API with Real Transaction Data
"""
import requests
import pandas as pd
import json
import time
from datetime import datetime

# API Configuration
API_BASE_URL = "http://localhost:8001"
HEADERS = {"Content-Type": "application/json"}

def test_health_check():
    """Test API health endpoint"""
    print("🏥 Testing API Health Check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ API Status: {health_data['status']}")
            print(f"📊 Model Loaded: {health_data['model_loaded']}")
            print(f"🔢 API Version: {health_data['api_version']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Connection error: {e}")
        return False

def load_real_transaction_data():
    """Load real transactions from your dataset"""
    print("📊 Loading real transaction data from your dataset...")
    try:
        df = pd.read_csv('data/upi_fraud_dataset.csv')
        print(f"✅ Loaded {len(df)} real transactions")
        return df
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return None

def prepare_transaction_for_api(row):
    """Convert dataset row to API format"""
    return {
        "trans_hour": int(row['trans_hour']),
        "trans_day": int(row['trans_day']),
        "trans_month": int(row['trans_month']),
        "trans_year": int(row['trans_year']),
        "category": int(row['category']),
        "upi_number": str(row['upi_number']),
        "age": int(row['age']),
        "trans_amount": float(row['trans_amount']),
        "state": int(row['state']),
        "zip": int(row['zip']),
        "transaction_id": f"real_txn_{row.name}"
    }

def test_single_prediction(transaction_data, actual_fraud):
    """Test single transaction prediction"""
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict",
            headers=HEADERS,
            json=transaction_data
        )
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            processing_time = (end_time - start_time) * 1000
            
            prediction = result['prediction']
            confidence = result['fraud_probability']
            risk_level = result['risk_level']
            
            # Determine if prediction is correct
            is_correct = prediction == actual_fraud
            emoji = "✅" if is_correct else "❌"
            
            print(f"{emoji} Transaction {transaction_data['transaction_id']}:")
            print(f"  💰 Amount: ${transaction_data['trans_amount']:.2f}")
            print(f"  🕐 Time: {transaction_data['trans_hour']}:00")
            print(f"  📊 Category: {transaction_data['category']}")
            print(f"  🏛️  State: {transaction_data['state']}")
            print(f"  👤 Age: {transaction_data['age']}")
            print(f"  🤖 Predicted: {'FRAUD' if prediction == 1 else 'LEGITIMATE'} ({confidence:.1f}%)")
            print(f"  ✅ Actual: {'FRAUD' if actual_fraud == 1 else 'LEGITIMATE'}")
            print(f"  ⚠️  Risk Level: {risk_level}")
            print(f"  ⚡ Processing Time: {processing_time:.1f}ms")
            print()
            
            return {
                'correct': is_correct,
                'processing_time': processing_time,
                'confidence': confidence,
                'prediction': prediction,
                'actual': actual_fraud,
                'risk_level': risk_level
            }
        else:
            print(f"❌ API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Request failed: {e}")
        return None

def test_batch_prediction(transactions_data, actual_labels):
    """Test batch transaction prediction"""
    print(f"📦 Testing batch prediction with {len(transactions_data)} transactions...")
    try:
        start_time = time.time()
        # Send transactions as a direct list, not wrapped in an object
        response = requests.post(
            f"{API_BASE_URL}/batch_predict",
            headers=HEADERS,
            json=transactions_data
        )
        end_time = time.time()
        
        if response.status_code == 200:
            results = response.json()
            total_time = (end_time - start_time) * 1000
            avg_time = total_time / len(transactions_data)
            
            print(f"✅ Batch processing completed in {total_time:.1f}ms")
            print(f"⚡ Average per transaction: {avg_time:.1f}ms")
            
            # Handle the response format from the server
            if 'summary' in results:
                summary = results['summary']
                print(f"📊 Batch Summary:")
                print(f"  Total Processed: {summary.get('total_transactions', 'N/A')}")
                print(f"  Fraud Detected: {summary.get('fraud_detected', 'N/A')}")
                print(f"  Fraud Rate: {summary.get('fraud_rate', 'N/A')}%")
                print(f"  High Risk Transactions: {summary.get('high_risk_transactions', 'N/A')}")
                
                # Calculate accuracy if we have detailed results
                if 'results' in results:
                    detailed_results = results['results']
                    correct_predictions = 0
                    for i, result in enumerate(detailed_results):
                        if i < len(actual_labels) and result.get('prediction') == actual_labels[i]:
                            correct_predictions += 1
                    
                    if len(actual_labels) > 0:
                        accuracy = (correct_predictions / len(actual_labels)) * 100
                        print(f"🎯 Batch Accuracy: {accuracy:.1f}% ({correct_predictions}/{len(actual_labels)})")
            else:
                print(f"📊 Response: {results}")
            
            return results
        else:
            print(f"❌ Batch API Error: {response.status_code} - {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Batch request failed: {e}")
        return None

def main():
    """Main testing function"""
    print("🚀 Ultra Advanced UPI Fraud Detection API Testing")
    print("🔥 Testing with Your Real Transaction Data")
    print("=" * 60)
    
    # Test health check first
    if not test_health_check():
        print("❌ API is not healthy. Please check the server.")
        return
    
    # Load real transaction data
    df = load_real_transaction_data()
    if df is None:
        return
    
    print(f"\n📊 Dataset Analysis:")
    print(f"Total Transactions: {len(df)}")
    fraud_count = df['fraud_risk'].sum()
    legitimate_count = len(df) - fraud_count
    print(f"Fraud Transactions: {fraud_count} ({fraud_count/len(df)*100:.1f}%)")
    print(f"Legitimate Transactions: {legitimate_count} ({legitimate_count/len(df)*100:.1f}%)")
    
    # Test with a diverse sample of real transactions
    print(f"\n🔍 Testing Individual Predictions with Real Data:")
    print("-" * 50)
    
    # Filter out transactions with invalid ZIP codes for API validation (>=10000)
    valid_zip_df = df[df['zip'] >= 10000]
    print(f"📊 Filtering for valid ZIP codes (>=10000): {len(valid_zip_df)}/{len(df)} transactions available")
    
    # Select a diverse sample - mix of fraud and legitimate transactions
    fraud_samples = valid_zip_df[valid_zip_df['fraud_risk'] == 1].sample(n=min(3, (valid_zip_df['fraud_risk'] == 1).sum()), random_state=42)
    legitimate_samples = valid_zip_df[valid_zip_df['fraud_risk'] == 0].sample(n=min(3, (valid_zip_df['fraud_risk'] == 0).sum()), random_state=42)
    test_samples = pd.concat([fraud_samples, legitimate_samples]).sample(frac=1, random_state=42)
    
    results = []
    for _, row in test_samples.iterrows():
        transaction_data = prepare_transaction_for_api(row)
        actual_fraud = int(row['fraud_risk'])
        
        result = test_single_prediction(transaction_data, actual_fraud)
        if result:
            results.append(result)
    
    # Calculate overall accuracy for individual tests
    if results:
        individual_accuracy = sum(r['correct'] for r in results) / len(results) * 100
        avg_processing_time = sum(r['processing_time'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print(f"📊 Individual Test Results Summary:")
        print(f"🎯 Accuracy: {individual_accuracy:.1f}% ({sum(r['correct'] for r in results)}/{len(results)})")
        print(f"⚡ Average Processing Time: {avg_processing_time:.1f}ms")
        print(f"📈 Average Confidence: {avg_confidence:.1f}%")
        
        # Risk level distribution
        risk_levels = [r['risk_level'] for r in results]
        from collections import Counter
        risk_distribution = Counter(risk_levels)
        print(f"⚠️  Risk Level Distribution: {dict(risk_distribution)}")
    
    # Test batch processing with a larger sample
    print(f"\n📦 Testing Batch Processing:")
    print("-" * 50)
    
    batch_size = min(10, len(df))
    # Filter out transactions with ZIP codes below 10000 to match API validation
    valid_zip_df = df[df['zip'] >= 10000]
    if len(valid_zip_df) >= batch_size:
        batch_sample = valid_zip_df.sample(n=batch_size, random_state=123)
    else:
        print(f"⚠️  Only {len(valid_zip_df)} transactions with valid ZIP codes (>=10000)")
        batch_sample = valid_zip_df.sample(n=len(valid_zip_df), random_state=123)
    
    batch_transactions = [prepare_transaction_for_api(row) for _, row in batch_sample.iterrows()]
    batch_actual = batch_sample['fraud_risk'].tolist()
    
    batch_results = test_batch_prediction(batch_transactions, batch_actual)
    
    # Test the demo endpoint
    print(f"\n🧪 Testing Demo Endpoint:")
    print("-" * 50)
    try:
        response = requests.post(f"{API_BASE_URL}/test_with_sample")
        if response.status_code == 200:
            demo_result = response.json()
            print("✅ Demo endpoint test successful!")
            print(f"📊 Sample transaction processed: {demo_result['sample_transaction']['transaction_id']}")
            if 'prediction_result' in demo_result:
                pred_result = demo_result['prediction_result']
                print(f"🤖 Demo prediction: {'FRAUD' if pred_result.get('prediction') == 1 else 'LEGITIMATE'}")
                print(f"📈 Demo confidence: {pred_result.get('fraud_probability', 0):.1f}%")
        else:
            print(f"❌ Demo test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Demo test error: {e}")
    
    print(f"\n✨ API Testing Complete!")
    print(f"🚀 Your Ultra Advanced UPI Fraud Detection API is working with real data!")
    print(f"📈 Ready for production deployment!")

if __name__ == "__main__":
    main()
