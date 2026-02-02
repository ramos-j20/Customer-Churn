#!/bin/bash
# ============================================
# PostgreSQL Initialization Script
# Creates databases for Airflow and MLflow
# ============================================

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create Airflow database
    CREATE DATABASE airflow_db;
    GRANT ALL PRIVILEGES ON DATABASE airflow_db TO $POSTGRES_USER;

    -- Create MLflow database  
    CREATE DATABASE mlflow_db;
    GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO $POSTGRES_USER;

    -- Log success
    \echo 'Databases airflow_db and mlflow_db created successfully!'
EOSQL
