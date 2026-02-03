FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (required for confluent-kafka)
RUN apt-get update && apt-get install -y \
    gcc \
    librdkafka-dev \
    netcat-openbsd \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY churn-features/ churn-features/
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir ./churn-features

# Copy source code
COPY src/ src/
COPY data/ data/

# Set Python path
ENV PYTHONPATH=/app/src

# Default command (can be overridden)
CMD ["python", "src/ingestion/producer.py"]
