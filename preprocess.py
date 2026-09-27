import pandas as pd
import numpy as np

# Extracted from the notebook statistics for scaling
SCALER_MEANS = {
    'tenure': 32.371149,
    'MonthlyCharges': 64.761692,
    'TotalCharges': 2279.734304
}

SCALER_STDS = {
    'tenure': 24.559481,
    'MonthlyCharges': 30.090047,
    'TotalCharges': 2266.794470
}

MODEL_FEATURES = [
    'SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges', 'gender_Male',
    'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'Contract_1.0', 'Contract_2.0', 'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check', 'TotalServices', 'ExpectedTotal',
    'ChargeDifference'
]

def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Transforms the raw input dictionary into a pandas DataFrame containing
    exactly the features required by the trained model.
    """
    
    # 1. Initialize output dictionary with 0s for all expected features
    out = {f: 0.0 for f in MODEL_FEATURES}
    
    # 2. Pass through numerical values (and scale them)
    # Handle potentially missing/empty TotalCharges
    total_charges_raw = data.get('TotalCharges', 0.0)
    if total_charges_raw == "" or total_charges_raw == " ":
        total_charges_raw = 0.0
    else:
        total_charges_raw = float(total_charges_raw)
        
    tenure_raw = float(data.get('tenure', 0.0))
    monthly_charges_raw = float(data.get('MonthlyCharges', 0.0))
    
    # Scaling
    tenure_scaled = (tenure_raw - SCALER_MEANS['tenure']) / SCALER_STDS['tenure']
    monthly_charges_scaled = (monthly_charges_raw - SCALER_MEANS['MonthlyCharges']) / SCALER_STDS['MonthlyCharges']
    total_charges_scaled = (total_charges_raw - SCALER_MEANS['TotalCharges']) / SCALER_STDS['TotalCharges']
    
    out['tenure'] = tenure_scaled
    out['MonthlyCharges'] = monthly_charges_scaled
    out['TotalCharges'] = total_charges_scaled
    
    # 3. Simple categorical mapping
    out['SeniorCitizen'] = float(data.get('SeniorCitizen', 0))
    
    if data.get('gender') == 'Male': out['gender_Male'] = 1.0
    if data.get('Partner') == 'Yes': out['Partner_Yes'] = 1.0
    if data.get('Dependents') == 'Yes': out['Dependents_Yes'] = 1.0
    if data.get('PhoneService') == 'Yes': out['PhoneService_Yes'] = 1.0
    if data.get('PaperlessBilling') == 'Yes': out['PaperlessBilling_Yes'] = 1.0
    
    # MultipleLines
    if data.get('MultipleLines') == 'No phone service':
        out['MultipleLines_No phone service'] = 1.0
    elif data.get('MultipleLines') == 'Yes':
        out['MultipleLines_Yes'] = 1.0
        
    # InternetService
    if data.get('InternetService') == 'Fiber optic':
        out['InternetService_Fiber optic'] = 1.0
    elif data.get('InternetService') == 'No':
        out['InternetService_No'] = 1.0
        
    # OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    total_services_count = 0
    
    for srv in services:
        val = data.get(srv)
        if val == 'No internet service':
            out[f'{srv}_No internet service'] = 1.0
        elif val == 'Yes':
            out[f'{srv}_Yes'] = 1.0
            total_services_count += 1
            
    # Contract (Ordinal mapping to 0, 1, 2, then one-hot)
    contract_val = data.get('Contract')
    if contract_val == 'One year':
        out['Contract_1.0'] = 1.0
    elif contract_val == 'Two year':
        out['Contract_2.0'] = 1.0
        
    # PaymentMethod
    pm_val = data.get('PaymentMethod')
    if pm_val == 'Credit card (automatic)':
        out['PaymentMethod_Credit card (automatic)'] = 1.0
    elif pm_val == 'Electronic check':
        out['PaymentMethod_Electronic check'] = 1.0
    elif pm_val == 'Mailed check':
        out['PaymentMethod_Mailed check'] = 1.0
        
    # 4. Computed Features
    out['TotalServices'] = float(total_services_count)
    out['ExpectedTotal'] = out['TotalServices'] * out['MonthlyCharges']
    out['ChargeDifference'] = out['TotalCharges'] - out['ExpectedTotal']
    
    # Create DataFrame ensuring correct column order
    df = pd.DataFrame([out])[MODEL_FEATURES]
    return df
