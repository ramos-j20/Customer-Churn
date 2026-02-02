import os
import time
import json
import random
import logging
import pandas as pd
from datetime import datetime
from confluent_kafka import Producer
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('churn_producer')

# Load environment variables
load_dotenv()

# Kafka config
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092')
TOPIC = 'live_churn_stream'

def delivery_report(err, msg):
    """Callback for delivery reports"""
    if err is not None:
        logger.error(f'Message delivery failed: {err}')
    else:
        # logger.debug(f'Message delivered to {msg.topic()} [{msg.partition()}]')
        pass

def load_data(filepath):
    """Load and validate CSV data"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    # Fill missing values if any
    df = df.fillna(0)
    logger.info(f"Loaded {len(df)} records from {filepath}")
    return df

def inject_noise(record):
    """Add random noise to numerical fields to simulate new data"""
    # Clone record to avoid mutating original
    data = record.copy()
    
    # +/- 5% noise for MonthlyCharges and TotalCharges
    noise_factor = random.uniform(0.95, 1.05)
    
    if 'MonthlyCharges' in data:
        data['MonthlyCharges'] = round(data['MonthlyCharges'] * noise_factor, 2)
        
    # TotalCharges is a string in the original CSV (sometimes empty), but if float:
    if 'TotalCharges' in data and isinstance(data['TotalCharges'], (int, float)):
         data['TotalCharges'] = round(data['TotalCharges'] * noise_factor, 2)
        
    return data

def main():
    # Initialize Producer
    conf = {
        'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
        'client.id': 'churn-producer-1',
    }
    producer = Producer(conf)
    
    # Load dataset
    data_path = os.path.join('data', 'customer_churn.csv')
    try:
        df = load_data(data_path)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    logger.info(f"Starting producer loop. Target topic: {TOPIC}")
    
    try:
        while True:
            # Iterate through dataframe
            for _, row in df.iterrows():
                # Convert to dict
                record = row.to_dict()
                
                # 1. Update timestamp to NOW (UTC)
                record['timestamp'] = datetime.utcnow().isoformat()
                
                # 2. Inject Noise
                record = inject_noise(record)
                
                # 3. Produce to Kafka
                try:
                    producer.produce(
                        TOPIC,
                        key=str(record.get('customerID', record.get('customer_id', 'unknown'))),
                        value=json.dumps(record),
                        callback=delivery_report
                    )
                    
                    # Serve delivery reports (callbacks)
                    producer.poll(0)
                    
                except BufferError:
                    logger.warning("Local buffer full, waiting...")
                    producer.poll(1)
                
                # 4. Simulate traffic
                time.sleep(0.5)
                
            logger.info("Finished one loop of dataset. Restarting...")
            producer.flush()
            
    except KeyboardInterrupt:
        logger.info("Stopping producer...")
    finally:
        producer.flush()

if __name__ == '__main__':
    main()
