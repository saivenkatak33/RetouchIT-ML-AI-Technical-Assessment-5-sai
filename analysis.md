os.makedirs("reports", exist_ok=True)

report = """
# Fraud Detection Analysis

### Key Preprocessing
- Used ColumnTransformer: scaled numeric features, one-hot encoded categorical `type`.
- Handled imbalance with:
  1. Algorithmic: class_weight='balanced'
  2. Resampling: SMOTE

### Model Comparison
- Logistic Regression, Random Forest, XGBoost compared with PR-AUC metric.
- XGBoost gave highest PR-AUC, selected as final.

### Business Tradeoffs
- False Positives (blocking legit transactions) cost 5x more than False Negatives.
- Threshold tuned to reduce false positives while keeping fraud recall reasonable.

### Explainability
- SHAP showed top 3 fraud-driving features:
  - `amount`
  - `oldbalanceOrg`
  - `type_TRANSFER`

### Conclusion
Final model: **XGBoost with SMOTE + preprocessing pipeline**, saved in `models/fraud_model.pkl`.  
Deep learning not chosen — explainability is more valuable than tiny accuracy gains.
"""

with open("reports/analysis.md", "w") as f:
    f.write(report)

print("✅ reports/analysis.md created")
