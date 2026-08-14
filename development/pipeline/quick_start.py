"""
Quick Start: MLflow + DVC Integration with MinIO
"""

import os
import subprocess
import sys
import time
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv


def ensure_bucket_exists(endpoint_url, access_key, secret_key, bucket_name):
    """Ensure MinIO bucket exists before DVC remote setup."""
    s3_client = boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )
    try:
        s3_client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' already exists.")
    except ClientError:
        print(f"📦 Bucket '{bucket_name}' not found. Creating bucket...")
        s3_client.create_bucket(Bucket=bucket_name)
        print(f"✅ Created bucket '{bucket_name}'.")


def quick_start():
    print("🚀 Quick Start: MLflow + DVC Integration with MinIO")
    print("=" * 50)

    # FIX: Point BASE_DIR to project root relative to this script file
    # quick_start.py lives in: <BASE_DIR>/development/pipeline/quick_start.py
    SCRIPT_DIR = Path(__file__).resolve().parent
    BASE_DIR = SCRIPT_DIR.parent.parent

    # Ensure dvc target directory exists before chdir
    dvc_dir = BASE_DIR / "development" / "dvc"
    dvc_dir.mkdir(parents=True, exist_ok=True)

    # Load environment variables from development/.env
    ENV_DIR = BASE_DIR / "development"
    load_dotenv(ENV_DIR / ".env")

    # 1. Install dependencies
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "mlflow",
            "dvc",
            "boto3",
            "catboost",
        ]
    )

    # 2. Read environment variables correctly
    s3_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    access_key = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    s3_bucket = os.getenv("MINIO_BUCKET_NAME", "mlflow-artifacts")

    print("⏳ Waiting for services...")
    time.sleep(5)

    # 3. Create bucket in MinIO if missing
    ensure_bucket_exists(s3_endpoint, access_key, secret_key, s3_bucket)

    # 4. Setup DVC with MinIO
    os.chdir(dvc_dir)

    subprocess.run(["dvc", "init", "--no-scm", "-f"])
    subprocess.run(["dvc", "remote", "add", "minio", f"s3://{s3_bucket}", "-f"])
    subprocess.run(["dvc", "remote", "modify", "minio", "endpointurl", s3_endpoint])
    subprocess.run(["dvc", "remote", "modify", "minio", "access_key_id", access_key])
    subprocess.run(["dvc", "remote", "modify", "minio", "secret_access_key", secret_key])
    subprocess.run(["dvc", "remote", "default", "minio"])

    # 5. Load models to MLflow
    os.chdir(BASE_DIR)
    subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "development" / "mlflow" / "load_existing_models.py"),
        ]
    )

    # 6. Run integration
    subprocess.run(
        [
            sys.executable,
            str(BASE_DIR / "development" / "pipeline" / "dvc_mlflow_integration.py"),
        ]
    )

    print("\n✅ Quick start complete!")
    print("\nNext steps:")
    print("1. MLflow UI: http://localhost:5003")
    print("2. MinIO Console: http://localhost:9001 (minioadmin/minioadmin)")
    print("3. DVC status: cd development/dvc && dvc status")


if __name__ == "__main__":
    quick_start()
