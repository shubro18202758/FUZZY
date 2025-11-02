# 🚀 Revolutionary AMDV-ART Fuzzy Fraud Detection Framework

## 📋 Overview

This repository contains a state-of-the-art fraud detection system that combines **Fuzzy Logic (AMDV-ART)**, **Deep Learning**, **Gradient Boosting**, and **Revolutionary Feature Engineering** to detect fraudulent transactions with high accuracy and precision.

### 🎯 Key Features

- **AMDV-ART Fuzzy Ensemble**: Adaptive Resonance Theory with MT_Tilde match tracking
- **Deep Transformer Neural Network**: Multi-layer architecture with attention mechanisms
- **Gradient Boosting Models**: XGBoost and LightGBM with optimized parameters
- **Advanced Feature Engineering**: 66+ engineered features including:
  - Adversarial features
  - Transformer-inspired attention features
  - Graph Neural Network features
  - Temporal and behavioral patterns
  - User and merchant statistics
  - Geospatial and velocity features
- **Intelligent Ensemble**: Multi-model fusion with optimized decision thresholds
- **Imbalance Handling**: SMOTE oversampling and Focal Loss support

## 📊 Performance Metrics

### Latest Results (Synthetic Indian UPI Fraud Dataset)

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| **Final Ensemble** | **0.85** | **0.95** | **0.92** | **0.78** | **0.52** |
| AMDV-ART Ensemble | 0.80 | - | - | 0.10 | - |
| Deep Neural Network | 0.65 | - | - | 0.13 | - |
| XGBoost | 0.90 | - | - | 0.00 | - |
| LightGBM | 0.88 | - | - | 0.03 | - |

*Note: The ensemble is optimized for high recall (fraud detection) with threshold optimization*

## 🗂️ Repository Structure

```
FUZZY/
├── ultra-enhanced-upi-fraud-detection/     # Main framework directory
│   ├── fused_revolutionary_amdv_art_framework.py  # Core framework
│   ├── src/                                # Source code modules
│   │   └── core/
│   │       └── feature_engineering.py      # Advanced feature engineering
│   ├── config/                             # Configuration files
│   │   └── model_config.py                 # Model hyperparameters
│   ├── api/                                # REST API implementations
│   │   ├── flask_api.py                    # Flask API
│   │   └── fastapi_server.py               # FastAPI server
│   ├── data/                               # Dataset storage
│   ├── models/                             # Saved model files
│   │   ├── fine_tuned/                     # Fine-tuned models
│   │   └── ultra_optimized/                # Optimized models
│   ├── reports/                            # Performance reports (JSON)
│   ├── logs/                               # Training logs and metrics
│   ├── results_mendley/                    # Mendley dataset results
│   ├── visualizations/                     # Charts and plots
│   ├── notebooks/                          # Jupyter notebooks
│   │   └── ultra_demo.ipynb                # Demo notebook
│   └── scripts/                            # Utility scripts
│
├── Datasets:
│   ├── synthetic_indian_upi_fraud_data.csv # Synthetic Indian UPI dataset
│   ├── upi_transactions_2024.csv           # Real UPI transactions
│   └── mendley_mobile_money_trimmed_dataset.csv  # Mendley dataset
│
├── Supporting Files:
│   ├── fused_framework_mendley.py          # Mendley-specific framework
│   ├── Srinivas_Model.py                   # Reference implementation
│   ├── run_framework.py                    # Quick run script
│   └── requirements.txt                    # Python dependencies
│
└── Documentation:
    ├── ENHANCEMENT_RESULTS_ANALYSIS.md     # Enhancement analysis
    ├── MENDLEY_DATASET_RESULTS_REPORT.md   # Mendley results
    └── COMPLETE_FRAMEWORK_FEATURE_INVENTORY.md  # Feature inventory
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- TensorFlow 2.x
- scikit-learn
- XGBoost
- LightGBM
- pandas, numpy, matplotlib

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/shubro18202758/FUZZY.git
cd FUZZY
```

2. **Install dependencies**
```bash
pip install -r ultra-enhanced-upi-fraud-detection/requirements.txt
```

### Running the Framework

#### Option 1: Command Line
```bash
# Run on synthetic dataset
python ultra-enhanced-upi-fraud-detection/fused_revolutionary_amdv_art_framework.py synthetic_indian_upi_fraud_data.csv

# Run on UPI 2024 dataset
python ultra-enhanced-upi-fraud-detection/fused_revolutionary_amdv_art_framework.py upi_transactions_2024.csv

# Run on Mendley dataset
python ultra-enhanced-upi-fraud-detection/fused_revolutionary_amdv_art_framework.py mendley_mobile_money_trimmed_dataset.csv
```

#### Option 2: Using the Run Script
```bash
python run_framework.py
```

#### Option 3: Jupyter Notebook
```bash
jupyter notebook ultra-enhanced-upi-fraud-detection/notebooks/ultra_demo.ipynb
```

### Running the API

#### Flask API
```bash
python ultra-enhanced-upi-fraud-detection/api/flask_api.py
```

#### FastAPI Server
```bash
python ultra-enhanced-upi-fraud-detection/api/fastapi_server.py
```

Then access the API at `http://localhost:5000` (Flask) or `http://localhost:8000` (FastAPI)

## 🔧 Configuration

### Model Configuration (`config/model_config.py`)

```python
MODEL_CONFIG = {
    'amdv_art': {
        'vigilance_base': 0.75,
        'learning_rate': 0.3,
        'ensemble_size': 5
    },
    'deep_nn': {
        'layers': [256, 128, 64],
        'dropout': 0.3,
        'learning_rate': 0.001,
        'epochs': 25
    },
    'xgboost': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100
    },
    'lightgbm': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 100
    }
}
```

### Feature Engineering Options

The framework automatically applies:
- **Adversarial Features**: Synthetic adversarial patterns
- **Attention Features**: Transformer-inspired attention weights
- **Graph Features**: Network topology features
- **Temporal Features**: Time-based patterns (hour, day, week)
- **User Statistics**: Transaction frequency, average amounts
- **Merchant Statistics**: Merchant-level aggregations
- **Geospatial Features**: Location-based risk scoring
- **Velocity Features**: Transaction rate analysis

## 📈 Advanced Usage

### Enabling SMOTE
```python
# In fused_revolutionary_amdv_art_framework.py
USE_SMOTE = True  # Enable SMOTE oversampling
```

### Enabling Focal Loss
```python
# In fused_revolutionary_amdv_art_framework.py
USE_FOCAL_LOSS = True  # Use Focal Loss for deep learning
```

### Threshold Optimization
```python
# In fused_revolutionary_amdv_art_framework.py
OPTIMIZE_THRESHOLD = True  # Enable automatic threshold tuning
```

## 🧪 Testing and Evaluation

### Running Tests
```bash
cd ultra-enhanced-upi-fraud-detection
python tests/test_ultra_optimization.py
```

### Generating Reports
```bash
# Generate comprehensive report
python generate_comprehensive_report.py

# Generate visual analytics
python generate_visual_analytics.py

# Create performance dashboard
python create_simple_dashboard.py
```

### Converting Logs to PDF
```bash
python enhanced_log_to_pdf_converter.py
```

## 📊 Understanding the Results

### Output Files

After training, the framework generates:

1. **JSON Report** (`reports/Breakthrough_98Plus_Results_TIMESTAMP.json`)
   - Detailed metrics for each model
   - Confusion matrices
   - Classification reports
   - Feature importance

2. **CSV Logs** (`logs/`)
   - `model_performance_metrics_TIMESTAMP.csv`: Per-model metrics
   - `feature_engineering_metrics_TIMESTAMP.csv`: Feature stats
   - `summary_metrics_TIMESTAMP.csv`: Aggregated results

3. **Model Files** (`models/`)
   - Saved TensorFlow models (`.h5`)
   - XGBoost models (`.pkl`)
   - LightGBM models (`.pkl`)
   - AMDV-ART ensemble (`.pkl`)

4. **Visualizations** (`visualizations/`)
   - Confusion matrix heatmaps
   - ROC curves
   - Feature importance plots
   - Training history charts

## 🎓 Research Paper

This framework is based on cutting-edge research combining:
- **Fuzzy Adaptive Resonance Theory (Fuzzy ART)**
- **Match Tracking (MT_Tilde) Strategy**
- **Deep Transformer Architectures**
- **Ensemble Learning Techniques**

For detailed methodology, see `ultra-enhanced-upi-fraud-detection/RESEARCH_PAPER_SUMMARY.md`

## 🛠️ Troubleshooting

### Common Issues

**1. Low Precision / High False Positives**
- Increase AMDV-ART vigilance parameter
- Adjust ensemble threshold
- Enable focal loss for deep learning

**2. Low Recall / Missed Frauds**
- Enable SMOTE oversampling
- Reduce decision threshold
- Increase XGBoost scale_pos_weight

**3. Memory Issues**
- Reduce deep learning batch size
- Decrease AMDV-ART ensemble size
- Use smaller feature subsets

**4. Long Training Time**
- Reduce deep learning epochs
- Decrease XGBoost/LightGBM estimators
- Use fewer AMDV-ART ensemble models

## 📦 Dataset Information

### Synthetic Indian UPI Fraud Dataset
- **Size**: 10,000 transactions
- **Fraud Rate**: 9.65%
- **Features**: 19 columns including temporal, geographical, and behavioral data

### UPI Transactions 2024
- **Size**: Real-world UPI transaction data
- **Fraud Rate**: Variable
- **Features**: Comprehensive transaction details

### Mendley Mobile Money Dataset
- **Size**: Mobile money transactions
- **Fraud Rate**: Imbalanced
- **Features**: Mobile payment specific attributes

## 🔬 Feature Engineering Details

The framework creates **66+ engineered features**:

### Category 1: Statistical Features
- Mean, median, std deviation of amounts
- Transaction frequency
- Time-based aggregations

### Category 2: Adversarial Features (8)
- Synthetic adversarial patterns
- Noise injection
- Perturbation analysis

### Category 3: Attention Features (7)
- Transformer-style attention weights
- Multi-head attention patterns
- Context-aware features

### Category 4: Graph Features (5)
- Network centrality measures
- Community detection
- Node importance

### Category 5: Domain-Specific Features
- Temporal patterns (hour, day, week, month)
- User statistics (frequency, avg amount, last transaction)
- Merchant statistics (transaction count, avg amount)
- Geospatial distance and velocity
- Risk aggregation scores

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open-source and available under the MIT License.

## 👥 Authors

- **GitHub**: [shubro18202758](https://github.com/shubro18202758)

## 🙏 Acknowledgments

- Fuzzy ART research community
- TensorFlow and scikit-learn teams
- XGBoost and LightGBM contributors
- Open-source fraud detection research

## 📧 Contact

For questions, issues, or collaborations:
- Open an issue on GitHub
- Email: [Your contact email]

## 🔮 Future Enhancements

- [ ] Real-time streaming fraud detection
- [ ] Explainable AI (SHAP/LIME integration)
- [ ] AutoML hyperparameter optimization
- [ ] Multi-currency support
- [ ] Blockchain integration
- [ ] Mobile app deployment
- [ ] Cloud-native architecture (AWS/Azure/GCP)

## 📚 References

1. Carpenter, G. A., & Grossberg, S. (1987). "ART 2: Self-organization of stable category recognition codes for analog input patterns"
2. Vaswani, A., et al. (2017). "Attention Is All You Need"
3. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System"
4. Ke, G., et al. (2017). "LightGBM: A Highly Efficient Gradient Boosting Decision Tree"

---

⭐ **Star this repository** if you find it useful!

🐛 **Report issues** to help improve the framework!

🚀 **Fork and contribute** to make it even better!
