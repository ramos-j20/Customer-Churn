import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.exceptions import NotFittedError

# constants
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

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply basic data cleaning logic."""
    df = df.copy()
    
    # 1. Handle TotalCharges (Empty strings -> 0)
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    
    # 2. Encode Churn if present (Training only)
    if 'Churn' in df.columns:
            # Just map it for safety if this DF is used for y extraction later outside
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
        
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features."""
    df = df.copy()
    
    # Ensure numerical columns are numeric
    for col in ['tenure', 'MonthlyCharges', 'TotalCharges']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
    # 2. Charges per tenure
    if 'TotalCharges' in df.columns and 'tenure' in df.columns:
        df['charges_per_tenure'] = df['TotalCharges'] / (df['tenure'] + 1)
    
    # 3. Service count
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    present_service_cols = [c for c in service_cols if c in df.columns]
    
    if present_service_cols:
        df['service_count'] = df[present_service_cols].apply(
            lambda row: sum(1 for x in row if x == 'Yes'), axis=1
        )
    else:
        df['service_count'] = 0
        
    # 4. Has any add-on service
    if 'service_count' in df.columns:
        df['has_addons'] = (df['service_count'] > 0).astype(int)
    
    # 5. Has premium support
    if 'TechSupport' in df.columns:
        df['has_premium_support'] = (df['TechSupport'] == 'Yes').astype(int)
    else:
            df['has_premium_support'] = 0
    
    # 6. Contract type numeric
    if 'Contract' in df.columns:
        contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
        df['contract_stability'] = df['Contract'].map(contract_map).fillna(0)
    
    # 7. High value customer
    GLOBAL_MEDIAN_CHARGES = 70.35
    if 'MonthlyCharges' in df.columns:
        df['is_high_value'] = (df['MonthlyCharges'] > GLOBAL_MEDIAN_CHARGES).astype(int)
    else:
        df['is_high_value'] = 0

    # 8. Automatic payment
    if 'PaymentMethod' in df.columns:
        df['auto_payment'] = df['PaymentMethod'].apply(
            lambda x: 1 if isinstance(x, str) and 'automatic' in x.lower() else 0
        )
    else:
        df['auto_payment'] = 0

    # 9. Senior citizen with dependents
    if all(col in df.columns for col in ['SeniorCitizen', 'Partner', 'Dependents']):
        df['senior_alone'] = ((df['SeniorCitizen'] == 1) & 
                            (df['Partner'] == 'No') & 
                            (df['Dependents'] == 'No')).astype(int)
    else:
            if 'SeniorCitizen' in df.columns: # Partial fallback
                df['senior_alone'] = 0

    # 10. Monthly to Total ratio
    if 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
        df['monthly_total_ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)
        
    # Ensure all expected columns exist (fill 0 if missing from logic)
    # Note: Accessing global constants
    for col in NUMERICAL_FEATURES + BINARY_FEATURES:
        if col not in df.columns:
            df[col] = 0.0 # Default fallback
            
    return df

def preprocess_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Full preprocessing pipeline: Cleaning + Feature Engineering"""
    df = clean_data(df)
    df = engineer_features(df)
    return df

class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    # Constants exposed in class as well for backward compat or easy access
    CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
    NUMERICAL_FEATURES = NUMERICAL_FEATURES
    BINARY_FEATURES = BINARY_FEATURES

    def __init__(self):
        # We group binary with numerical for scaling, or we could pass them through.
        # Original code seemed to treat them as numeric features eventually.
        # We will scale everything numeric + binary to be safe for models like LR/SVM, 
        # though trees don't care.
        self.final_numeric_features = self.NUMERICAL_FEATURES + self.BINARY_FEATURES
        
        self.pipeline = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.final_numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), self.CATEGORICAL_FEATURES)
            ],
            remainder='drop'  # Drop extra columns (like 'Churn' or IDs)
        )

    def fit(self, X, y=None):
        """Fit the preprocessor to the data."""
        # Feature Engineering first to generate the columns we need to fit on
        X_engineered = preprocess_pipeline(X)
        self.pipeline.fit(X_engineered, y)
        return self

    def transform(self, X):
        """Transform the data using the fitted preprocessor."""
        X_engineered = preprocess_pipeline(X)
        try:
            return self.pipeline.transform(X_engineered)
        except NotFittedError as e:
            raise NotFittedError(f"ChurnPreprocessor is not fitted yet. Call 'fit' before transform.") from e

    def fit_transform(self, X, y=None):
        """Fit to data, then transform it."""
        X_engineered = preprocess_pipeline(X)
        return self.pipeline.fit_transform(X_engineered, y)
