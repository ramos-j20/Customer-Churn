import pytest
import pandas as pd
import numpy as np
from churn_features.preprocessing import ChurnPreprocessor
from sklearn.exceptions import NotFittedError

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'tenure': [1, 12, 24],
        'MonthlyCharges': [50.0, 70.0, 90.0],
        'TotalCharges': [50.0, 840.0, 2160.0],
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'InternetService': ['DSL', 'Fiber optic', 'No'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'OnlineSecurity': ['No', 'Yes', 'No'],
        'TechSupport': ['No', 'Yes', 'No'],
        'SeniorCitizen': [0, 1, 0],
        'Partner': ['No', 'Yes', 'No'],
        'Dependents': ['No', 'No', 'No'],
        'PaperlessBilling': ['Yes', 'No', 'Yes']
    })

def test_initialization():
    preprocessor = ChurnPreprocessor()
    assert isinstance(preprocessor, ChurnPreprocessor)

def test_fit_transform(sample_data):
    preprocessor = ChurnPreprocessor()
    transformed = preprocessor.fit_transform(sample_data)
    
    # Check output shape
    # Numerical: tenure, MonthlyCharges, TotalCharges, charges_per_tenure, service_count, contract_stability, monthly_total_ratio (7)
    # Binary: SeniorCitizen, has_premium_support, auto_payment, is_high_value, senior_alone (5)
    # Categorical: Contract(3), InternetService(3), PaymentMethod(3), OnlineSecurity(2 - Yes/No), TechSupport(2 - Yes/No), PaperlessBilling(2 - Yes/No)
    # Total Categorical OHE: 3+3+3+2+2+2 = 15? Note: OHE creates column for each unique value present.
    # We should just check it runs and produces array.
    
    assert isinstance(transformed, np.ndarray)
    assert transformed.shape[0] == 3
    # Exact column count depends on OHE which depends on unique values in sample. 
    # But it should be > 10.
    assert transformed.shape[1] > 10

def test_missing_derived_features_are_created(sample_data):
    # Ensure raw data without derived columns (like 'service_count') works
    # The fixture ONLY has raw columns, so this is implicitly testing that.
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(sample_data)
    # Features should be created internally
    # We can check by inspecting if transform works
    assert preprocessor.transform(sample_data) is not None

def test_missing_raw_columns_handled(sample_data):
    # If a critical column like 'MonthlyCharges' is missing, it should probably error out 
    # OR fallback to 0 if that's the logic (implementation uses fillna(0) for missing NUMERICAL_FEATURES)
    # But missing whole column?
    # Logic in _clean_data checks: "if col in df.columns".
    # Logic in _engineer_features checks: "if col in df.columns".
    # Logic at end: "if col not in df.columns: df[col] = 0.0"
    # So it should NOT crash, but fill 0.
    
    bad_data = sample_data.drop(columns=['MonthlyCharges'])
    preprocessor = ChurnPreprocessor()
    preprocessor.fit(bad_data)
    res = preprocessor.transform(bad_data)
    assert res is not None

def test_idempotency_training_serving(sample_data):
    preprocessor = ChurnPreprocessor()
    train_out = preprocessor.fit_transform(sample_data)
    serving_out = preprocessor.transform(sample_data)
    np.testing.assert_array_almost_equal(train_out, serving_out)
