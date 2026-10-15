import boto3
import os
from pathlib import Path
from dotenv import load_dotenv

# Configuration
BASE_PATH = Path(__file__).parents[2]  # hospital_production
ENV_PATH = BASE_PATH / ".env"
load_dotenv(ENV_PATH)

# MinIO configuration
MINIO_ROOT_USER = os.getenv("MINIO_ROOT_USER")
MINIO_ROOT_PASSWORD = os.getenv("MINIO_ROOT_PASSWORD")
MINIO_BUCKET = 'hospital-models'
MLFLOW_S3_ENDPOINT_URL= os.getenv("MLFLOW_S3_ENDPOINT_URL")

# Create a MinIO bucket if it doesn't exist
def ensure_minio_bucket():
    # Create a MinIO client
    minio_client = boto3.client(
        "s3",
        endpoint_url=MLFLOW_S3_ENDPOINT_URL,
        aws_access_key_id=MINIO_ROOT_USER,
        aws_secret_access_key=MINIO_ROOT_PASSWORD,
    )

    # Create the bucket
    minio_client.create_bucket(Bucket='hospital-models')

# Usage
if __name__ == "__main__":
    ensure_minio_bucket()
    print(f"Ensured MinIO bucket '{MINIO_BUCKET}' exists.")