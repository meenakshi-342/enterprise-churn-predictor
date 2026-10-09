import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATA_PATH = "data/telco_churn.csv"
CLEANED_DATA_PATH = "data/cleaned_telco_churn.csv"


def preprocess_data():
    print("Loading dataset for preprocessing...")
    df = pd.read_csv(DATA_PATH)

    # 1. Drop identifier column
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # 2. Fix numeric data types
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    # 3. Separate target variable
    target_col = "Churn"
    df[target_col] = df[target_col].map({"Yes": 1, "No": 0})

    # 4. Encode categorical features
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 5. Save cleaned dataset
    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned dataset saved to '{CLEANED_DATA_PATH}'")

    print("\n--- Cleaned Dataset Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Target Distribution (Churn rate):\n{df['Churn'].value_counts(normalize=True)}")
    print("\nFirst 5 Rows of Cleaned Data:")
    print(df.head())

    return df


if __name__ == "__main__":
    preprocess_data()