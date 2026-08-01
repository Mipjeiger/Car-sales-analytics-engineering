"""
Load models from MLflow to DVC-tracked directory with MinIO
"""

import mlflow
import shutil
import json
import os
from dotenv import load_dotenv
import boto3
from pathlib import Path
from botocore.client import Config

def load_models_to_dvc():
    BASE_DIR = Path.cwd()
    DVC_DATA_DIR = BASE_DIR / 'development' / 'dvc' / 'data'
    DVC_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ENV_DIR = BASE_DIR / 'development' / '.env'
    if ENV_DIR.exists():
        load_dotenv(dotenv_path=ENV_DIR)
    
    # Setup MinIO client
    s3_client = boto3.client(
        's3',
        endpoint_url='http://minio:9000',
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )
    
    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    client = mlflow.tracking.MlflowClient()
    models = client.search_registered_models()
    
    for model in models:
        latest = client.get_latest_versions(model.name, stages=["Production"])
        if latest:
            run = client.get_run(latest[0].run_id)
            artifact_uri = run.info.artifact_uri
            
            if artifact_uri.startswith('s3://'):
                path = artifact_uri.replace('s3://mlflow-artifacts/', '')
                local_path = DVC_DATA_DIR / model.name
                local_path.mkdir(parents=True, exist_ok=True)
                
                # Download from MinIO
                paginator = s3_client.get_paginator('list_objects_v2')
                for page in paginator.paginate(Bucket='mlflow-artifacts', Prefix=path):
                    for obj in page.get('Contents', []):
                        s3_key = obj['Key']
                        relative_path = s3_key.replace(path, '').lstrip('/')
                        if relative_path:
                            target = local_path / relative_path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            s3_client.download_file('mlflow-artifacts', s3_key, str(target))
            
            print(f"✅ Loaded {model.name} from MinIO to DVC")

if __name__ == "__main__":
    load_models_to_dvc()