import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

CLEANED_DATA_PATH = "data/cleaned_telco_churn.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_churn_model.pkl")


def train_models():
    print("Loading cleaned dataset...")
    df = pd.read_csv(CLEANED_DATA_PATH)

    # Separate features (X) and target (y)
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Split dataset into Training (80%) and Testing (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}\n")

    # Initialize models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=42
        ),
        "XGBoost": XGBClassifier(
            eval_metric="logloss", random_state=42
        ),
    }

    best_model = None
    best_auc = 0.0
    best_model_name = ""

    # Train and evaluate models
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"=== {name} ===")
        print(f"Accuracy : {acc:.4f}")
        print(f"ROC-AUC  : {auc:.4f}\n")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_model_name = name

    # Save the best performing model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    print("========================================")
    print(f"🏆 Best Model: {best_model_name} (ROC-AUC: {best_auc:.4f})")
    print(f"Saved best model to '{MODEL_PATH}'")
    print("========================================")


if __name__ == "__main__":
    train_models()