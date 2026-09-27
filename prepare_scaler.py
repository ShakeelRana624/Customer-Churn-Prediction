import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

try:
    path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    df = pd.read_csv(path + "/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    df = df.drop('customerID', axis=1)

    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    scaler.fit(X_train[numeric_cols])

    joblib.dump(scaler, 'c:/Users/shake/OneDrive/Desktop/Customer-Churn-Perdiction/scaler.pkl')
    print("Scaler saved successfully.")
except Exception as e:
    print(f"Error: {e}")
