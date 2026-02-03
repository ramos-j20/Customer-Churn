import os
import json
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from io import BytesIO
from minio import Minio
from confluent_kafka import Consumer, KafkaError
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('s3_sink')

# Load environment variables
load_dotenv()

# Config
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
TOPIC = 'live_churn_stream'
GROUP_ID = 'churn-data-lake-sink'
# MinIO Config
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ROOT_USER')
MINIO_SECRET_KEY = os.getenv('MINIO_ROOT_PASSWORD')
BUCKET_NAME = 'churn-lake'

# Batch Config
BATCH_SIZE = 50
BATCH_TIMEOUT = 30.0  # seconds (simple implementation relies on size primarily)

def get_minio_client():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

def upload_batch(minio_client, batch_data):
    """Convert batch to Parquet and upload to MinIO"""
    if not batch_data:
        return

    try:
        # Convert to DataFrame
        df = pd.DataFrame(batch_data)
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)
        
        # Write to in-memory buffer
        buffer = BytesIO()
        pq.write_table(table, buffer)
        buffer.seek(0)
        file_size = buffer.getbuffer().nbytes
        
        # Generate S3 Path
        # Structure: raw/year=YYYY/month=MM/timestamp_batchId.parquet
        now = datetime.utcnow()
        year = now.year
        month = f"{now.month:02d}"
        filename = f"raw/year={year}/month={month}/{int(now.timestamp())}_batch.parquet"
        
        # Upload
        minio_client.put_object(
            BUCKET_NAME,
            filename,
            buffer,
            file_size,
            content_type='application/octet-stream'
        )
        
        logger.info(f"Uploaded batch to {BUCKET_NAME}/{filename} ({len(batch_data)} records)")
        
    except Exception as e:
        logger.error(f"Failed to upload batch: {e}")

def main():
    # Initialize MinIO
    minio_client = get_minio_client()
    
    # Initialize Consumer
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'group.id': GROUP_ID,
        'auto.offset.reset': 'earliest'
    }
    consumer = Consumer(conf)
    consumer.subscribe([TOPIC])
    
    logger.info(f"Starting consumer group {GROUP_ID} on topic {TOPIC}")
    
    batch_buffer = []

    try:
        while True:
            msg = consumer.poll(1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                else:
                    logger.error(f"Consumer error: {msg.error()}")
                    continue

            # Process message
            try:
                record = json.loads(msg.value().decode('utf-8'))
                batch_buffer.append(record)
                
                # Check batch size
                if len(batch_buffer) >= BATCH_SIZE:
                    upload_batch(minio_client, batch_buffer)
                    batch_buffer = []  # Clear buffer
                    
            except json.JSONDecodeError:
                logger.error(f"Failed to decode message: {msg.value()}")
                continue

    except KeyboardInterrupt:
        logger.info("Stopping consumer...")
    finally:
        # Upload remaining records
        if batch_buffer:
            logger.info("Uploading remaining records...")
            upload_batch(minio_client, batch_buffer)
            
        consumer.close()

if __name__ == '__main__':
    main()
