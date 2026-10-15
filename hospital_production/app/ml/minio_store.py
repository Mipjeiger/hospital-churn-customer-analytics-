import io
import json
import joblib
import logging
from app.api.config import settings
from minio import Minio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""MinIO artifact storage utility functions"""

def _client():
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ROOT_USER,
        secret_key=settings.MINIO_ROOT_PASSWORD,
        secure=False
    )

def ensure_bucket():
    c = _client()
    if not c.bucket_exists(settings.MINIO_BUCKET):

        c.make_bucket(settings.MINIO_BUCKET)
        logger.info(f"Created bucket: {settings.MINIO_BUCKET}")

def save_artifact(key: str, obj) -> str:
    """Save any picklable object to Minio as .joblib file"""
    ensure_bucket()
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    buf.seek(0)
    path = f"{key}.joblib"
    _client().put_object(settings.MINIO_BUCKET, path, buf, length=buf.getbuffer().nbytes, content_type="application/octet-stream")
    return f"s3://{settings.MINIO_BUCKET}/{path}"

def load_artifact(key: str):
    path = f"{key}.joblib"
    r = _client().get_object(settings.MINIO_BUCKET, path)

    try:
        return joblib.load(io.BytesIO(r.read()))
    finally:
        r.close()
        r.release_conn()

def save_json(key: str, data: dict):
    ensure_bucket()
    body = json.dumps(data, indent=2, default=str).encode()
    _client().put_object(settings.MINIO_BUCKET, f"{key}.json", io.BytesIO(body), length=len(body), content_type="application/json")