os.makedirs("models", exist_ok=True)

model_code = """
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import average_precision_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from preprocessing import build_preprocessor

def train_and_save(X_train, y_train, X_test, y_test):
    preprocessor = build_preprocessor()

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(class_weight='balanced'),
        'XGBClassifier': XGBClassifier(scale_pos_weight=100, use_label_encoder=False, eval_metric='logloss')
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        pipe = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

        pipe.fit(X_train, y_train)
        preds = pipe.predict_proba(X_test)[:,1]
        score = average_precision_score(y_test, preds)

        if score > best_score:
            best_score = score
            best_model = pipe

    joblib.dump(best_model, "models/fraud_model.pkl")
    print(f"Best model saved with PR-AUC: {best_score}")
"""

with open("src/model_comparison.py", "w") as f:
    f.write(model_code)

print("✅ src/model_comparison.py created")
