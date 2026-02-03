# MLflow with PostgreSQL support
FROM ghcr.io/mlflow/mlflow:v2.9.2

# Install dependencies for healthcheck and database
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir psycopg2-binary
