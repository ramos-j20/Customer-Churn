import os
import io
import pandas as pd
from minio import Minio
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
MINIO_ENDPOINT = os.getenv('MINIO_ENDPOINT', 'localhost:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ROOT_USER', 'minioadmin')
MINIO_SECRET_KEY = os.getenv('MINIO_ROOT_PASSWORD', 'minioadmin')
BUCKET_NAME = 'churn-lake'

def get_minio_client():
    return Minio(
        MINIO_ENDPOINT.replace('http://', ''),
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

def list_files(client, bucket):
    print(f"Listing files in bucket '{bucket}'...")
    objects = client.list_objects(bucket, recursive=True)
    files = [obj.object_name for obj in objects]
    return sorted(files, reverse=True)

def read_parquet(client, bucket, filename):
    print(f"\nReading file: {filename}")
    try:
        response = client.get_object(bucket, filename)
        data = response.read()
        response.close()
        response.release_conn()
        
        df = pd.read_parquet(io.BytesIO(data))
        return df
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    try:
        client = get_minio_client()
        
        if not client.bucket_exists(BUCKET_NAME):
            print(f"Bucket '{BUCKET_NAME}' does not exist.")
            return

        files = list_files(client, BUCKET_NAME)
        
        if not files:
            print("No files found in the bucket.")
            return

        print(f"Found {len(files)} files.")
        print("Latest 5 files:")
        for i, f in enumerate(files[:5]):
            print(f"{i+1}. {f}")

        # Read the latest file automatically
        latest_file = files[0]
        df = read_parquet(client, BUCKET_NAME, latest_file)
        
        if df is not None:
            print(f"\nData Shape: {df.shape}")
            print("\nColumns:")
            print(df.columns.tolist())
            print("\nFirst 5 rows:")
            print(df.head().to_string())
            print("\nData Types:")
            print(df.dtypes)
            
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
