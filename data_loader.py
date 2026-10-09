import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
DATA_FILE = os.path.join(DATA_DIR, "telco_churn.csv")


def generate_local_dataset(num_samples=7000):
    """Generates a realistic Telco Customer Churn dataset locally."""
    np.random.seed(42)

    customer_ids = [f"{i:04d}-TELCO" for i in range(1, num_samples + 1)]
    genders = np.random.choice(["Male", "Female"], size=num_samples)
    senior_citizens = np.random.choice([0, 1], size=num_samples, p=[0.84, 0.16])
    partners = np.random.choice(["Yes", "No"], size=num_samples, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], size=num_samples, p=[0.30, 0.70])

    tenure = np.random.randint(1, 73, size=num_samples)
    phone_service = np.random.choice(
        ["Yes", "No"], size=num_samples, p=[0.90, 0.10]
    )
    multiple_lines = np.random.choice(
        ["No phone service", "No", "Yes"], size=num_samples, p=[0.10, 0.48, 0.42]
    )
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"], size=num_samples, p=[0.34, 0.44, 0.22]
    )

    online_security = np.random.choice(
        ["Yes", "No", "No internet service"],
        size=num_samples,
        p=[0.28, 0.50, 0.22],
    )
    online_backup = np.random.choice(
        ["Yes", "No", "No internet service"],
        size=num_samples,
        p=[0.34, 0.44, 0.22],
    )
    device_protection = np.random.choice(
        ["Yes", "No", "No internet service"],
        size=num_samples,
        p=[0.34, 0.44, 0.22],
    )
    tech_support = np.random.choice(
        ["Yes", "No", "No internet service"],
        size=num_samples,
        p=[0.29, 0.49, 0.22],
    )
    streaming_tv = np.random.choice(
        ["Yes", "No", "No internet service"],
        size=num_samples,
        p=[0.38, 0.40, 0.22],
    )
    streaming_movies = np.random.choice(
        ["Yes", "No", "No internet service"],
        size=num_samples,
        p=[0.39, 0.39, 0.22],
    )

    contract = np.random.choice(
        ["Month-to-month", "One year", "Two year"],
        size=num_samples,
        p=[0.55, 0.21, 0.24],
    )
    paperless_billing = np.random.choice(
        ["Yes", "No"], size=num_samples, p=[0.59, 0.41]
    )
    payment_method = np.random.choice(
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
        size=num_samples,
        p=[0.33, 0.23, 0.22, 0.22],
    )

    monthly_charges = np.round(
        np.random.uniform(18.25, 118.75, size=num_samples), 2
    )
    total_charges = np.round(
        monthly_charges * tenure + np.random.uniform(-10, 10, size=num_samples), 2
    )
    total_charges = np.where(total_charges < 0, monthly_charges, total_charges)

    # Churn probabilities based on tenure & contract
    churn_prob = (
        0.45
        - (tenure / 100.0)
        + (contract == "Month-to-month") * 0.25
        - (contract == "Two year") * 0.20
    )
    churn_prob = np.clip(churn_prob, 0.05, 0.85)
    churn = np.random.binomial(1, churn_prob)
    churn = np.where(churn == 1, "Yes", "No")

    df = pd.DataFrame(
        {
            "customerID": customer_ids,
            "gender": genders,
            "SeniorCitizen": senior_citizens,
            "Partner": partners,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "Churn": churn,
        }
    )

    return df


def load_dataset():
    """Loads local dataset or creates it locally."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if not os.path.exists(DATA_FILE):
        print("Generating local Telco Customer Churn dataset...")
        df = generate_local_dataset()
        df.to_csv(DATA_FILE, index=False)
        print(f"Dataset successfully saved to '{DATA_FILE}'")
    else:
        print(f"Dataset found locally at '{DATA_FILE}'")
        df = pd.read_csv(DATA_FILE)

    print("\n--- Success! Dataset Overview ---")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
    print("\nFirst 5 Rows:")
    print(df.head(5))

    return df


if __name__ == "__main__":
    load_dataset()