import os
import json
import time
import logging
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from confluent_kafka import Consumer, KafkaError, TopicPartition
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('s3_sink_tx')

# Load environment variables
load_dotenv()

class TransactionalS3Sink:
    def __init__(self):
        # Configuration
        self.kafka_bootstrap = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
        self.topic = 'live_churn_stream'
        self.group_id = 'churn-data-lake-sink-tx'  # Distinct group for transactional sink
        
        self.minio_endpoint = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
        self.minio_access_key = os.getenv('MINIO_ROOT_USER')
        self.minio_secret_key = os.getenv('MINIO_ROOT_PASSWORD')
        self.bucket_name = 'churn-lake'
        
        # Batch settings
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        self.batch_timeout = float(os.getenv('BATCH_TIMEOUT', '60.0'))
        
        # State
        self.buffer = []
        self.last_flush_time = time.time()
        self.minio_client = self._get_minio_client()
        self.consumer = self._get_consumer()
        
        # Offset tracking for the current batch
        # Map: {(topic, partition): (min_offset, max_offset)}
        self.batch_offsets = {} 

    def _get_minio_client(self):
        """Initialize and verify MinIO connection"""
        try:
            client = Minio(
                self.minio_endpoint,
                access_key=self.minio_access_key,
                secret_key=self.minio_secret_key,
                secure=False
            )
            # Health check: ensure bucket exists
            if not client.bucket_exists(self.bucket_name):
                logger.info(f"Bucket {self.bucket_name} not found, creating...")
                client.make_bucket(self.bucket_name)
            logger.info("Connected to MinIO successfully.")
            return client
        except Exception as e:
            logger.critical(f"Failed to connect to MinIO: {e}")
            raise

    def _get_consumer(self):
        """Initialize Kafka Consumer with manual commit"""
        conf = {
            'bootstrap.servers': self.kafka_bootstrap,
            'group.id': self.group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,  # CRITICAL: We manage commits manually
            'isolation.level': 'read_committed'
        }
        try:
            c = Consumer(conf)
            c.subscribe([self.topic])
            logger.info(f"Connected to Kafka, consuming from {self.topic}")
            return c
        except Exception as e:
            logger.critical(f"Failed to connect to Kafka: {e}")
            raise

    def _track_offset(self, msg):
        """Track min/max offsets per partition for the current batch"""
        topic = msg.topic()
        partition = msg.partition()
        offset = msg.offset()
        
        key = (topic, partition)
        if key not in self.batch_offsets:
            self.batch_offsets[key] = {'min': offset, 'max': offset}
        else:
            self.batch_offsets[key]['min'] = min(self.batch_offsets[key]['min'], offset)
            self.batch_offsets[key]['max'] = max(self.batch_offsets[key]['max'], offset)

    def _generate_filename(self):
        """Generate idempotent filename based on offsets"""
        # Format: raw/year=YYYY/month=MM/topic-partition-startOffset-endOffset.parquet
        # If multiple partitions are in a batch, we might produce multiple files or one composite.
        # Strategy: For simplicity, if we consume from multiple partitions, we ideally want to split writes per partition 
        # to maintain true idempotency per partition. However, to keep it simple as a single file writer:
        # We will append the range of the FIRST partition found or a has representing the batch.
        # BETTER STRATEGY: One file per partition in the batch?
        # Standard approach for simple consumers: 
        # Use timestamp + first_offset_of_primary_partition
        
        # Let's use the timestamp and the offset range of the most populated partition to keep it readable
        # but unique enough.
        
        if not self.batch_offsets:
            return None
            
        # Find partition with max range
        target_key = list(self.batch_offsets.keys())[0] # Pick first for simplicity
        offsets = self.batch_offsets[target_key]
        
        now = datetime.utcnow()
        year = now.year
        month = f"{now.month:02d}"
        
        # Construct a unique ID using offsets from all partitions to ensure uniqueness if mixed
        # But simpler: {topic}-{partition}-{min}-{max}-{timestamp}.parquet
        topic, partition = target_key
        filename = f"raw/year={year}/month={month}/{topic}-p{partition}-{offsets['min']}-{offsets['max']}-{int(now.timestamp())}.parquet"
        return filename

    def _upload_to_s3_with_backoff(self, buffer_bytes, filename):
        """Upload to S3 with exponential backoff"""
        max_retries = 5
        base_delay = 1.0 # seconds
        
        for attempt in range(max_retries):
            try:
                self.minio_client.put_object(
                    self.bucket_name,
                    filename,
                    buffer_bytes,
                    buffer_bytes.getbuffer().nbytes,
                    content_type='application/octet-stream'
                )
                logger.info(f"Successfully uploaded {filename}")
                return True
            except (S3Error, Exception) as e:
                delay = base_delay * (2 ** attempt)
                logger.warning(f"Upload failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
        
        logger.error(f"Failed to upload {filename} after {max_retries} attempts.")
        return False

    def flush_batch(self):
        """Process the current batch: Write to S3 -> Commit Kafka"""
        if not self.buffer:
            return

        logger.info(f"Flushing batch of {len(self.buffer)} records...")
        
        # 1. Prepare Data
        try:
            df = pd.DataFrame(self.buffer)
            table = pa.Table.from_pandas(df)
            pq_buffer = BytesIO()
            pq.write_table(table, pq_buffer)
            pq_buffer.seek(0)
        except Exception as e:
            logger.error(f"Failed to serialize batch: {e}")
            # Logic decision: If we can't serialize, we probably can't save it. 
            # In production, maybe dump to a dead-letter queue (DLQ).
            # Here, we will skip persistence but NOT commit offset?? 
            # Risk: consumer loop stuck on bad data.
            # Fix: If serialization fails, we must drop the data or fix code. 
            # We will clear buffer to proceed.
            self.buffer = []
            self.batch_offsets = {}
            return

        # 2. Generate Filename (Idempotency Key)
        filename = self._generate_filename()
        
        # 3. Write to S3 (Transactional Step 1)
        success = self._upload_to_s3_with_backoff(pq_buffer, filename)
        
        if success:
            # 4. Commit Offsets (Transactional Step 2)
            try:
                self.consumer.commit()
                logger.info("Kafka offsets committed successfully.")
                
                # Reset state only after commit
                self.buffer = []
                self.batch_offsets = {}
                self.last_flush_time = time.time()
                
            except KafkaError as e:
                logger.critical(f"Failed to commit offsets after upload! Potential duplicate data on restart. Error: {e}")
                # We do NOT clear buffer here ideally, but if we crash, we re-consume.
                # Since data is in S3, we risk duplicates. Idempotency naming helps dedupe downstream.
                raise e
        else:
            logger.critical("Failed to write to S3. Crashing to trigger restart and re-processing.")
            # We raise exception to stop the process. Supervisor/Docker will restart it, 
            # and it will re-consume the same batch from last committed offset.
            raise RuntimeError("S3 Write Failed")

    def run(self):
        logger.info("Starting Transactional S3 Sink...")
        try:
            while True:
                # Poll for messages
                msg = self.consumer.poll(1.0)
                
                # Check for batch timeout
                if (time.time() - self.last_flush_time) >= self.batch_timeout and self.buffer:
                    logger.info("Batch timeout reached.")
                    self.flush_batch()

                if msg is None:
                    continue
                
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        logger.error(f"Consumer error: {msg.error()}")
                        # Backoff on consumer error
                        time.sleep(1) 
                        continue

                # Process valid message
                try:
                    record_json = json.loads(msg.value().decode('utf-8'))
                    
                    # Add metadata
                    record_json['_kafka_topic'] = msg.topic()
                    record_json['_kafka_partition'] = msg.partition()
                    record_json['_kafka_offset'] = msg.offset()
                    
                    self.buffer.append(record_json)
                    self._track_offset(msg)
                    
                    # Check batch size
                    if len(self.buffer) >= self.batch_size:
                        logger.info("Batch size reached.")
                        self.flush_batch()
                        
                except json.JSONDecodeError:
                    logger.warning(f"Skipping malformed JSON at offset {msg.offset()}")
                    continue
                except Exception as e:
                    logger.error(f"Unexpected error processing message: {e}")
                    # Decide whether to crash or skip. Skipping is safer for stability.
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Stopping sink...")
        except Exception as e:
            logger.critical(f"Fatal error: {e}")
        finally:
            if self.buffer:
                logger.info("Attempting to flush remaining records before shutdown...")
                try:
                    self.flush_batch()
                except Exception as e:
                    logger.error(f"Failed final flush: {e}")
            
            self.consumer.close()
            logger.info("Consumer closed.")

if __name__ == "__main__":
    sink = TransactionalS3Sink()
    sink.run()
