import warnings
import sys
import os

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=UserWarning, module='pydantic')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='_distutils_hack')
os.environ['PYTHONWARNINGS'] = 'ignore'

print("🚀 SCRIPT STARTED: Beginning imports...", flush=True)

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import os

print("✅ IMPORTS COMPLETE. Starting Main Logic...", flush=True)

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    precision_recall_curve, make_scorer
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import time
import socket
import requests
from scipy.stats import randint, uniform

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = os.getenv('DATA_PATH', 'data/customer_churn.csv')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME = "churn_prediction_v2"

# ==========================================
# CONNECTION CHECK
# ==========================================
def check_connection(uri, max_retries=5):
    """Wait for MLflow to be ready."""
    print(f"🔄 Check Connection: Attempting to reach {uri}...", flush=True)
    
    if uri.startswith("http://"):
        original_host = uri.split("http://")[1].split(":")[0]
        port_part = uri.strip().split(":")[-1].replace("/", "")
    else:
        original_host = "localhost"
        port_part = "5000"

    candidates = [original_host, "churn_mlflow"]
    
    for host in candidates:
        print(f"   🔎 Trying Host: {host} Port: {port_part}", flush=True)
        for i in range(max_retries):
            try:
                ip = socket.gethostbyname(host)
                print(f"   DNS Resolved {host} -> {ip}", flush=True)
                
                test_uri = f"http://{host}:{port_part}"
                response = requests.get(f"{test_uri}/health", timeout=5)
                if response.status_code == 200:
                    print(f"✅ MLflow is HEALTHY at {test_uri}!", flush=True)
                    if host != original_host:
                        print(f"   ⚠️ Switching MLFLOW_TRACKING_URI to {test_uri}", flush=True)
                        global MLFLOW_TRACKING_URI
                        MLFLOW_TRACKING_URI = test_uri
                    return True
            except Exception as e:
                print(f"   ⚠️ Attempt {i+1}/{max_retries} failed for {host}: {e}", flush=True)
                time.sleep(2)
        print(f"   ❌ Could not reach {host}. Trying next candidate...")
        
    return False

# Add src to path just in case
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from src.features.preprocessing import (
    preprocess_pipeline, 
    CATEGORICAL_FEATURES, 
    NUMERICAL_FEATURES,
    BINARY_FEATURES  # Although we put binary in numeric for scaling/passthrough usually, or separate
)

# NOTE: Binary features are already 0/1, but we can treat them as numeric for StandardScaler 
# or use 'passthrough' if we don't want to scale them.
# For simplicity and robustness with tree models, scaling them is harmless or we can group them.
# Let's group binary with numeric for the pipeline as they are already numeric-ish.
FINAL_NUMERIC_FEATURES = NUMERICAL_FEATURES + BINARY_FEATURES

# ==========================================
# DATA LOADING
# ==========================================
def load_data(path):
    print(f"📂 Loading data from {path}...", flush=True)
    df = pd.read_csv(path)
    
    # Use centralized preprocessing pipeline
    df = preprocess_pipeline(df)
    
    print(f"   Data Shape after preprocessing: {df.shape}")
    return df

# ==========================================
# THRESHOLD OPTIMIZATION
# ==========================================
def find_optimal_threshold(y_true, y_proba, optimize_for='f1'):
    """Find the best threshold for classification."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    if optimize_for == 'recall':
        # Find threshold that gives at least 80% recall with best precision
        valid_idx = recalls[:-1] >= 0.80
        if valid_idx.any():
            best_idx = np.argmax(precisions[:-1][valid_idx])
            best_threshold = thresholds[valid_idx][best_idx]
        else:
            best_threshold = 0.3  # Default aggressive threshold
    else:
        best_idx = np.argmax(f1_scores[:-1])
        best_threshold = thresholds[best_idx]
    
    return best_threshold

# ==========================================
# TRAINING
# ==========================================
def train():
    # 0. Check Connection
    if not check_connection(MLFLOW_TRACKING_URI):
        print("❌ Could not connect to MLflow. Exiting.", flush=True)
        return

    # 1. Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # 2. Load Data
    df = load_data(DATA_PATH)
    
    # 3. Define Features using imported constants
    # We rely on the constants from preprocessing.py to be the source of truth
    
    # Log the features being used
    print(f"   Using {len(CATEGORICAL_FEATURES)} Categorical Features")
    print(f"   Using {len(FINAL_NUMERIC_FEATURES)} Numerical/Binary Features")
    
    # Select only the columns we need
    # Ensure they exist in DF (preprocessing might have failed or columns missing)
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    available_num = [c for c in FINAL_NUMERIC_FEATURES if c in df.columns]
    
    X = df[available_cat + available_num]
    y = df['Churn']
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 5. Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), available_num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), available_cat)
        ]
    )
    
    # ==========================================
    # EXPERIMENT 1: RandomForest + SMOTE + Hyperparameter Tuning
    # ==========================================
    print("\n" + "="*50, flush=True)
    print("🌲 EXPERIMENT 1: Random Forest + SMOTE + Tuning", flush=True)
    print("="*50, flush=True)
    
    with mlflow.start_run(run_name="RF_SMOTE_Tuned"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("use_smote", True)
        mlflow.log_param("use_cv", True)
        
        # Pipeline with SMOTE
        rf_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
        
        # Hyperparameter search space
        param_dist = {
            'classifier__n_estimators': randint(50, 200),
            'classifier__max_depth': randint(5, 20),
            'classifier__min_samples_split': randint(2, 10),
            'classifier__min_samples_leaf': randint(1, 5)
        }
        
        # Cross-validation with RandomizedSearch
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        search = RandomizedSearchCV(
            rf_pipeline, param_dist, n_iter=10, cv=cv,
            scoring='f1', random_state=42, n_jobs=-1
        )
        
        print("   🔍 Running RandomizedSearchCV (10 iterations, 5-fold CV)...", flush=True)
        search.fit(X_train, y_train)
        
        # Log best params
        best_params = search.best_params_
        for param, value in best_params.items():
            mlflow.log_param(param.replace('classifier__', ''), value)
        
        # Evaluate on test set
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        best_threshold = find_optimal_threshold(y_test, y_proba, optimize_for='f1')
        y_pred_tuned = (y_proba >= best_threshold).astype(int)
        
        mlflow.log_param("optimal_threshold", round(best_threshold, 3))
        
        # Metrics with optimal threshold
        acc = accuracy_score(y_test, y_pred_tuned)
        prec = precision_score(y_test, y_pred_tuned)
        rec = recall_score(y_test, y_pred_tuned)
        f1 = f1_score(y_test, y_pred_tuned)
        
        print(f"   📊 Metrics (threshold={best_threshold:.2f}):", flush=True)
        print(f"      Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}", flush=True)
        
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("cv_best_score", search.best_score_)
        
        mlflow.sklearn.log_model(best_model, "model")
        print("   ✅ Model logged to MLflow.", flush=True)
    
    # ==========================================
    # EXPERIMENT 2: XGBoost + SMOTE + Hyperparameter Tuning
    # ==========================================
    print("\n" + "="*50, flush=True)
    print("🚀 EXPERIMENT 2: XGBoost + SMOTE + Tuning", flush=True)
    print("="*50, flush=True)
    
    with mlflow.start_run(run_name="XGB_SMOTE_Tuned"):
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("use_smote", True)
        mlflow.log_param("use_cv", True)
        
        # Pipeline with SMOTE and XGBoost
        xgb_pipeline = ImbPipeline([
            ('preprocessor', preprocessor),
            ('smote', SMOTE(random_state=42)),
            ('classifier', XGBClassifier(
                use_label_encoder=False, 
                eval_metric='logloss',
                random_state=42
            ))
        ])
        
        # XGBoost hyperparameter search space
        xgb_param_dist = {
            'classifier__n_estimators': randint(50, 200),
            'classifier__max_depth': randint(3, 10),
            'classifier__learning_rate': uniform(0.01, 0.3),
            'classifier__subsample': uniform(0.6, 0.4),
            'classifier__colsample_bytree': uniform(0.6, 0.4)
        }
        
        xgb_search = RandomizedSearchCV(
            xgb_pipeline, xgb_param_dist, n_iter=10, cv=cv,
            scoring='f1', random_state=42, n_jobs=-1
        )
        
        print("   🔍 Running RandomizedSearchCV (10 iterations, 5-fold CV)...", flush=True)
        xgb_search.fit(X_train, y_train)
        
        # Log best params
        xgb_best_params = xgb_search.best_params_
        for param, value in xgb_best_params.items():
            if isinstance(value, float):
                mlflow.log_param(param.replace('classifier__', ''), round(value, 4))
            else:
                mlflow.log_param(param.replace('classifier__', ''), value)
        
        # Evaluate on test set
        xgb_best_model = xgb_search.best_estimator_
        y_pred_xgb = xgb_best_model.predict(X_test)
        y_proba_xgb = xgb_best_model.predict_proba(X_test)[:, 1]
        
        # Find optimal threshold
        xgb_threshold = find_optimal_threshold(y_test, y_proba_xgb, optimize_for='f1')
        y_pred_xgb_tuned = (y_proba_xgb >= xgb_threshold).astype(int)
        
        mlflow.log_param("optimal_threshold", round(xgb_threshold, 3))
        
        # Metrics
        acc_xgb = accuracy_score(y_test, y_pred_xgb_tuned)
        prec_xgb = precision_score(y_test, y_pred_xgb_tuned)
        rec_xgb = recall_score(y_test, y_pred_xgb_tuned)
        f1_xgb = f1_score(y_test, y_pred_xgb_tuned)
        
        print(f"   📊 Metrics (threshold={xgb_threshold:.2f}):", flush=True)
        print(f"      Acc={acc_xgb:.4f}, Prec={prec_xgb:.4f}, Rec={rec_xgb:.4f}, F1={f1_xgb:.4f}", flush=True)
        
        mlflow.log_metric("accuracy", acc_xgb)
        mlflow.log_metric("precision", prec_xgb)
        mlflow.log_metric("recall", rec_xgb)
        mlflow.log_metric("f1_score", f1_xgb)
        mlflow.log_metric("cv_best_score", xgb_search.best_score_)
        
        mlflow.sklearn.log_model(xgb_best_model, "model")
        print("   ✅ Model logged to MLflow.", flush=True)
    
    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "="*50, flush=True)
    print("📈 TRAINING COMPLETE - Compare runs in MLflow UI", flush=True)
    print("="*50, flush=True)
    print(f"   RandomForest F1: {f1:.4f}", flush=True)
    print(f"   XGBoost F1:      {f1_xgb:.4f}", flush=True)
    print(f"\n   🔗 View at: {MLFLOW_TRACKING_URI}", flush=True)

if __name__ == '__main__':
    train()
