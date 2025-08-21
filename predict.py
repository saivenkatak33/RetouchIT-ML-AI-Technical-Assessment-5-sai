import pandas as pd
import joblib

# 1️⃣ Load trained pipeline
model = joblib.load("models/fraud_model.pkl")
print("✅ Model loaded successfully.")

# 2️⃣ Load new data
data = pd.read_csv("datatransactions_3000.csv")  # replace with any new CSV

# 3️⃣ Keep only required columns
required_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
if "isFraud" in data.columns:
    X = data[required_columns]
else:
    X = data[required_columns]

# 4️⃣ Make predictions
preds = model.predict(X)
probs = model.predict_proba(X)[:, 1]

# 5️⃣ Add predictions to dataset
data["prediction"] = preds
data["fraud_probability"] = probs

# 6️⃣ Save results
data.to_csv("predictions_output.csv", index=False)
print("✅ Predictions saved in predictions_output.csv")

# ---- Single transaction example ----
new_transaction = pd.DataFrame([{
    "type": "TRANSFER",
    "amount": 10000.0,
    "oldbalanceOrg": 50000.0,
    "newbalanceOrig": 40000.0,
    "oldbalanceDest": 100000.0,
    "newbalanceDest": 110000.0
}])

X_new = new_transaction[required_columns]

pred_prob = model.predict_proba(X_new)[:, 1][0]
pred_label = model.predict(X_new)[0]

print("\n--- Single Transaction Prediction ---")
print("Fraud probability:", round(pred_prob, 4))
print("Predicted label (0=legit, 1=fraud):", pred_label)
