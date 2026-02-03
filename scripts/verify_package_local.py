import sys
import os
import pandas as pd
import numpy as np

# Add the package to path so we can import it without installing
sys.path.append(os.path.join(os.getcwd(), 'churn-features', 'src'))

try:
    from churn_features.preprocessing import ChurnPreprocessor
    print("✅ Successfully imported ChurnPreprocessor")
except ImportError as e:
    print(f"❌ Failed to import ChurnPreprocessor: {e}")
    sys.exit(1)

def run_verification():
    print("🚀 Starting Verification...")

    # 1. Create Sample Data (replicating raw input format)
    df = pd.DataFrame({
        'tenure': [1, 12, 72],
        'MonthlyCharges': [29.85, 56.95, 118.75],
        'TotalCharges': ["29.85", "1889.50", "8600.00"], # String format to test cleaning
        'Contract': ['Month-to-month', 'One year', 'Two year'],
        'InternetService': ['DSL', 'DSL', 'Fiber optic'],
        'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        'OnlineSecurity': ['No', 'Yes', 'Yes'],
        'TechSupport': ['No', 'Yes', 'Yes'],
        'SeniorCitizen': [0, 0, 1],
        'Partner': ['Yes', 'No', 'Yes'],
        'Dependents': ['No', 'No', 'No'],
        'PaperlessBilling': ['Yes', 'No', 'Yes']
    })
    
    print(f"   Input Data Shape: {df.shape}")

    # 2. Initialize Preprocessor
    processor = ChurnPreprocessor()
    
    # 3. Fit and Transform
    print("   Running fit_transform...")
    try:
        X_out = processor.fit_transform(df)
        print(f"✅ Transformation Successful!")
        print(f"   Output Shape: {X_out.shape}")
        
        # Verify logical correctness
        # Check if output is numeric
        if np.issubdtype(X_out.dtype, np.number):
            print("   Output is numeric: YES")
        else:
            print("   Output is numeric: NO")
            
    except Exception as e:
        print(f"❌ Transformation Failed: {e}")
        raise e

    print("🎉 Verification Complete!")

if __name__ == "__main__":
    run_verification()
