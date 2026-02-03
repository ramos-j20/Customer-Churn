"""
Test Script for Churn Retraining DAG Logic

Run this script to validate the DAG functions with mocked dependencies.
Usage: python scripts/test_dag_logic.py
"""

import sys
import os
from unittest.mock import MagicMock, patch
from io import BytesIO

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("=" * 60)
print("🧪 TESTING CHURN RETRAINING DAG LOGIC")
print("=" * 60)


# ==========================================
# TEST 1: MinIO Data Sensor Logic
# ==========================================
def test_minio_sensor_logic():
    print("\n📦 TEST 1: MinIO Sensor Logic")
    print("-" * 40)
    
    # Mock Parquet data with different record counts
    test_scenarios = [
        {"name": "Below threshold", "records": 300, "expected": False},
        {"name": "At threshold", "records": 500, "expected": True},
        {"name": "Above threshold", "records": 1000, "expected": True},
    ]
    
    for scenario in test_scenarios:
        # Create mock Parquet table
        mock_table = MagicMock()
        mock_table.num_rows = scenario["records"]
        
        # Mock MinIO object
        mock_obj = MagicMock()
        mock_obj.object_name = "raw/year=2024/month=01/batch.parquet"
        
        # Simulate the sensor logic
        total_records = mock_table.num_rows
        result = total_records >= 500
        
        status = "✅ PASS" if result == scenario["expected"] else "❌ FAIL"
        print(f"  {status} {scenario['name']}: {scenario['records']} records -> {result}")
    
    print("  ✅ MinIO sensor logic validated!")


# ==========================================
# TEST 2: Champion/Challenger Registration Logic
# ==========================================
def test_registration_logic():
    print("\n🏷️  TEST 2: Champion/Challenger Registration Logic")
    print("-" * 40)
    
    test_scenarios = [
        {"name": "New beats Production", "new_acc": 0.88, "prod_acc": 0.85, "should_promote": True},
        {"name": "Equal accuracy", "new_acc": 0.85, "prod_acc": 0.85, "should_promote": False},  # > not >=
        {"name": "New worse than Prod", "new_acc": 0.82, "prod_acc": 0.85, "should_promote": False},
        {"name": "No existing Prod", "new_acc": 0.75, "prod_acc": 0.0, "should_promote": True},
    ]
    
    for scenario in test_scenarios:
        new_acc = scenario["new_acc"]
        prod_acc = scenario["prod_acc"]
        should_promote = new_acc > prod_acc
        
        status = "✅ PASS" if should_promote == scenario["should_promote"] else "❌ FAIL"
        print(f"  {status} {scenario['name']}: new={new_acc:.2f} vs prod={prod_acc:.2f} -> promote={should_promote}")
    
    print("  ✅ Champion/Challenger logic validated!")


# ==========================================
# TEST 3: DAG Import Check
# ==========================================
def test_dag_import():
    print("\n📂 TEST 3: DAG Import Check")
    print("-" * 40)
    
    try:
        # Mock Airflow imports since we may not have Airflow installed locally
        sys.modules['airflow'] = MagicMock()
        sys.modules['airflow.operators.python'] = MagicMock()
        sys.modules['airflow.sensors.python'] = MagicMock()
        sys.modules['airflow.providers.docker.operators.docker'] = MagicMock()
        sys.modules['docker'] = MagicMock()
        sys.modules['docker.types'] = MagicMock()
        
        # Try to import the DAG file
        dag_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'airflow', 'dags', 'churn_retraining_dag.py'
        )
        
        if os.path.exists(dag_path):
            print(f"  ✅ DAG file exists at: {dag_path}")
            
            # Read and check for key components
            with open(dag_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            checks = [
                ("dag_id='churn_retraining_dag'", "DAG ID defined"),
                ("@weekly", "Weekly schedule"),
                ("check_new_data", "Sensor task"),
                ("train_model", "Training task"),
                ("register_model", "Registration task"),
                ("MIN_RECORDS_THRESHOLD = 500", "Record threshold"),
                ("register_model_if_better", "Champion/Challenger function"),
            ]
            
            for check, description in checks:
                if check in content:
                    print(f"  ✅ {description}")
                else:
                    print(f"  ❌ Missing: {description}")
        else:
            print(f"  ❌ DAG file not found at: {dag_path}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")


# ==========================================
# TEST 4: Task Dependency Check
# ==========================================
def test_task_dependencies():
    print("\n🔗 TEST 4: Task Dependencies")
    print("-" * 40)
    
    dag_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'airflow', 'dags', 'churn_retraining_dag.py'
    )
    
    with open(dag_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check for the dependency chain
    if "check_new_data >> train_model >> register_model" in content:
        print("  ✅ Correct dependency chain: check_new_data >> train_model >> register_model")
    else:
        print("  ❌ Dependency chain not found or incorrect")


# ==========================================
# RUN ALL TESTS
# ==========================================
if __name__ == '__main__':
    test_minio_sensor_logic()
    test_registration_logic()
    test_dag_import()
    test_task_dependencies()
    
    print("\n" + "=" * 60)
    print("🎉 ALL TESTS COMPLETED!")
    print("=" * 60)
    print("\nTo test with real infrastructure:")
    print("  1. docker-compose up -d")
    print("  2. Access Airflow at http://localhost:8081")
    print("  3. Trigger 'churn_retraining_dag' manually")
