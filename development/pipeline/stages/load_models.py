"""
Load models from MLflow registry to DVC-tracked directory via MinIO/S3
"""

import os
import json
import mlflow
import boto3
from pathlib import Path
from dotenv import load_dotenv
from botocore.client import Config


def load_models_to_dvc():
    # Resolve paths dynamically relative to script location
    # Script location: development/pipeline/stages/load_models.py
    SCRIPT_DIR = Path(__file__).resolve().parent      # development/pipeline/stages
    DEV_DIR = SCRIPT_DIR.parents[1]                   # development
    DVC_DATA_DIR = DEV_DIR / 'dvc' / 'data'
    DVC_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Load environment variables from development/.env
    env_path = DEV_DIR / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    # Dynamic Endpoints
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
    minio_endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")

    # Setup MLflow
    mlflow.set_tracking_uri(tracking_uri)
    print(f"📡 Connected to MLflow Tracking Server at: {tracking_uri}")

    # Setup MinIO / S3 Client
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    s3_client = boto3.client(
        's3',
        endpoint_url=minio_endpoint,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

    client = mlflow.tracking.MlflowClient()
    registered_models = client.search_registered_models()

    if not registered_models:
        print("⚠️ No registered models found in MLflow.")
        return

    loaded_count = 0
    for model in registered_models:
        # Search for all versions of this model regardless of stage
        versions = client.search_model_versions(f"name='{model.name}'")
        if not versions:
            continue

        # Get latest version numerically
        latest_version_obj = max(versions, key=lambda v: int(v.version))
        run = client.get_run(latest_version_obj.run_id)
        artifact_uri = run.info.artifact_uri

        if artifact_uri.startswith('s3://'):
            # Parse bucket and prefix path dynamically
            s3_path = artifact_uri.replace('s3://', '')
            parts = s3_path.split('/', 1)
            bucket_name = parts[0]
            prefix = parts[1] if len(parts) > 1 else ''

            local_path = DVC_DATA_DIR / model.name
            local_path.mkdir(parents=True, exist_ok=True)

            try:
                paginator = s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                    for obj in page.get('Contents', []):
                        s3_key = obj['Key']
                        relative_path = s3_key.replace(prefix, '').lstrip('/')
                        if relative_path:
                            target = local_path / relative_path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            s3_client.download_file(bucket_name, s3_key, str(target))
                print(f"✅ Loaded {model.name} (v{latest_version_obj.version}) from MinIO to DVC")
                loaded_count += 1
            except Exception as e:
                print(f"⚠️ Could not download artifacts for {model.name} from MinIO: {e}")

    print(f"🎉 Successfully loaded {loaded_count} model(s) into {DVC_DATA_DIR}")


if __name__ == "__main__":
    load_models_to_dvc()