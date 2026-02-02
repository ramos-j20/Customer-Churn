
import pandas as pd
import numpy as np

# ==========================================
# CONSTANTS: FEATURE SELECTION
# ==========================================
CATEGORICAL_FEATURES = [
    'Contract', 'InternetService', 'PaymentMethod', 
    'OnlineSecurity', 'TechSupport', 'PaperlessBilling'
]

NUMERICAL_FEATURES = [
    'tenure', 'MonthlyCharges', 'TotalCharges',
    'charges_per_tenure', 'service_count', 
    'contract_stability', 'monthly_total_ratio'
]

BINARY_FEATURES = [
    'SeniorCitizen', 'has_premium_support', 'auto_payment', 
    'is_high_value', 'senior_alone'
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES + BINARY_FEATURES


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic data cleaning logic.
    Works for both Batch (Training) and Real-time (Inference) DF.
    """
    df = df.copy()
    
    # 1. Handle TotalCharges (Empty strings -> 0)
    # Ensure it's treated as string first if mixed types
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # 2. Encode Churn if present (Training only)
    if 'Churn' in df.columns:
        # Handle "Yes"/"No" mapping robustly
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for model training and inference.
    Centralized logic to ensure consistency across pipeline.
    """
    df = df.copy()
    
    # Ensure numerical columns are numeric before math operations
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 1. Tenure Buckets (Categorical - optional, if needed for analysis)
    # Note: For tree models, raw 'tenure' is usually better than buckets.
    # We keep it implicitly by using raw tenure.
    
    # 2. Charges per tenure (value extraction rate)
    # Avoid division by zero
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # 3. Service count (how many add-ons they have)
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Filter columns that actually exist in the input features
    present_service_cols = [c for c in service_cols if c in df.columns]
    
    if present_service_cols:
        df['service_count'] = df[present_service_cols].apply(
            lambda row: sum(1 for x in row if x == 'Yes'), axis=1
        )
        
    # 4. Has any add-on service
    if 'service_count' in df.columns:
        df['has_addons'] = (df['service_count'] > 0).astype(int)
    
    # 5. Has premium support
    if 'TechSupport' in df.columns:
        df['has_premium_support'] = (df['TechSupport'] == 'Yes').astype(int)
    
    # 6. Contract type numeric (longer = more stable)
    if 'Contract' in df.columns:
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df['contract_stability'] = df['Contract'].map(contract_map).fillna(0)
    
    # 7. High value customer (above median charges)
    GLOBAL_MEDIAN_CHARGES = 70.35  # Approximate median from EDA
    if 'MonthlyCharges' in df.columns:
        df['is_high_value'] = (df['MonthlyCharges'] > GLOBAL_MEDIAN_CHARGES).astype(int)

    # 8. Automatic payment
    if 'PaymentMethod' in df.columns:
        df['auto_payment'] = df['PaymentMethod'].apply(
            lambda x: 1 if isinstance(x, str) and 'automatic' in x.lower() else 0
        )

    # 9. Senior citizen with dependents (vulnerable segment)
    if all(col in df.columns for col in ['SeniorCitizen', 'Partner', 'Dependents']):
        df['senior_alone'] = ((df['SeniorCitizen'] == 1) & 
                            (df['Partner'] == 'No') & 
                            (df['Dependents'] == 'No')).astype(int)

    # 10. Monthly to Total ratio (payment consistency indicator)
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['monthly_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
        
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: Cleaning + Feature Engineering"""
    df = clean_data(df)
    df = engineer_features(df)
    return df
