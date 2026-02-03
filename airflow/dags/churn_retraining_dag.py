"""
Airflow DAG for Automated Churn Model Retraining

This DAG runs weekly to:
1. Check MinIO for new streaming data (>500 records)
2. Trigger model retraining via Docker
3. Register model to Production if accuracy > 0.80
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.providers.docker.operators.docker import DockerOperator
from docker.types import Mount
import os
import logging

# ==========================================
# CONFIGURATION
# ==========================================
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ROOT_USER')
MINIO_SECRET_KEY = os.getenv('MINIO_ROOT_PASSWORD')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
BUCKET_NAME = 'churn-lake'
MIN_RECORDS_THRESHOLD = 500

logger = logging.getLogger(__name__)

# ==========================================
# SENSOR FUNCTION: Check MinIO for new data
# ==========================================
def check_minio_for_new_data(**context):
    """
    Check if MinIO has enough new Parquet files for retraining.
    Returns True if >500 records are available.
    """
    from minio import Minio
    import pyarrow.parquet as pq
    from io import BytesIO
    
    client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
    
    total_records = 0
    
    try:
        # List all Parquet files in raw/ prefix
        objects = client.list_objects(BUCKET_NAME, prefix='raw/', recursive=True)
        
        for obj in objects:
            if obj.object_name.endswith('.parquet'):
                # Download and count records
                response = client.get_object(BUCKET_NAME, obj.object_name)
                data = BytesIO(response.read())
                response.close()
                response.release_conn()
                
                table = pq.read_table(data)
                total_records += table.num_rows
                
                # Early exit if we have enough
                if total_records >= MIN_RECORDS_THRESHOLD:
                    logger.info(f"✅ Found {total_records} records - threshold met!")
                    return True
        
        logger.info(f"⏳ Only {total_records} records found, need {MIN_RECORDS_THRESHOLD}")
        return False
        
    except Exception as e:
        logger.error(f"❌ Error checking MinIO: {e}")
        return False


# ==========================================
# REGISTRATION FUNCTION: Compare and promote if better
# ==========================================
def register_model_if_better(**context):
    """
    Compare the new model against the current Production model.
    Promote only if the new model has better accuracy (champion/challenger).
    """
    import mlflow
    from mlflow.tracking import MlflowClient
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()
    
    try:
        # Get the latest run from churn experiment (the new model)
        experiment = client.get_experiment_by_name("churn_prediction_s3")
        if not experiment:
            logger.error("❌ Experiment 'churn_prediction_s3' not found")
            return False
        
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if not runs:
            logger.error("❌ No runs found in experiment")
            return False
        
        latest_run = runs[0]
        new_accuracy = latest_run.data.metrics.get('accuracy', 0)
        logger.info(f"📊 New model accuracy: {new_accuracy:.4f}")
        
        # Get current Production model's accuracy
        model_name = "churn_model_prod"
        current_prod_accuracy = 0.0
        
        try:
            prod_versions = client.get_latest_versions(model_name, stages=["Production"])
            if prod_versions:
                prod_version = prod_versions[0]
                prod_run = client.get_run(prod_version.run_id)
                current_prod_accuracy = prod_run.data.metrics.get('accuracy', 0)
                logger.info(f"📊 Current Production model accuracy: {current_prod_accuracy:.4f}")
            else:
                logger.info("ℹ️ No Production model exists yet - will promote new model")
        except Exception as e:
            logger.info(f"ℹ️ Could not get Production model: {e} - will promote new model")
        
        # Compare: promote only if new model is better
        if new_accuracy > current_prod_accuracy:
            versions = client.search_model_versions(f"name='{model_name}'")
            if versions:
                latest_version = max(versions, key=lambda v: int(v.version))
                
                client.transition_model_version_stage(
                    name=model_name,
                    version=latest_version.version,
                    stage="Production",
                    archive_existing_versions=True
                )
                
                improvement = new_accuracy - current_prod_accuracy
                logger.info(f"🚀 Model v{latest_version.version} promoted to Production!")
                logger.info(f"   Improvement: {current_prod_accuracy:.4f} → {new_accuracy:.4f} (+{improvement:.4f})")
                return True
            else:
                logger.warning("⚠️ No model versions found to promote")
                return False
        else:
            logger.info(f"⏸️ New model ({new_accuracy:.4f}) did NOT beat Production ({current_prod_accuracy:.4f})")
            logger.info("   Keeping current Production model.")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during model comparison: {e}")
        raise


# ==========================================
# DAG DEFINITION
# ==========================================
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='churn_retraining_dag',
    default_args=default_args,
    description='Weekly churn model retraining with data validation and model registration',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['ml', 'churn', 'retraining'],
) as dag:
    
    # Task 1: Check MinIO for sufficient new data
    check_new_data = PythonSensor(
        task_id='check_new_data',
        python_callable=check_minio_for_new_data,
        poke_interval=60,  # Check every minute
        timeout=600,  # 10 minute timeout
        mode='reschedule',  # Free worker while waiting
    )
    
    # Task 2: Run training via Docker
    train_model = DockerOperator(
        task_id='train_model',
        image='customer-churn-trainer',  # Built from project Dockerfile
        command='python src/models/train.py',
        network_mode='churn_network',
        environment={
            'MLFLOW_TRACKING_URI': 'http://mlflow:5000',
            # Point to S3 Data Lake (Parquet folder)
            'DATA_PATH': 's3://churn-lake/raw/',
            'MLFLOW_S3_ENDPOINT_URL': 'http://minio:9000',
            'AWS_ACCESS_KEY_ID': MINIO_ACCESS_KEY,
            'AWS_SECRET_ACCESS_KEY': MINIO_SECRET_KEY,
            'GIT_PYTHON_REFRESH': 'quiet',
        },
        mounts=[
            Mount(source='/app/src', target='/app/src', type='bind'),
            # No need to mount /app/data anymore as we read from S3
        ],
        auto_remove=True,
        docker_url='unix://var/run/docker.sock',
    )
    
    # Task 3: Compare & register model if better than production
    register_model = PythonOperator(
        task_id='register_model',
        python_callable=register_model_if_better,
    )
    
    # Task Dependencies
    check_new_data >> train_model >> register_model
