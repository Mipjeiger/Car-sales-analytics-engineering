import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import joblib
import json
import io
import boto3
from dotenv import load_dotenv
from botocore.client import Config
from datetime import datetime
import os

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / 'database' / 'car_sales_prediction_sales.parquet'
ENV_DIR = BASE_DIR / '.env'
load_dotenv(dotenv_path=ENV_DIR)

# MinIO Configuration
MINIO_ENDPOINT = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
BUCKET_NAME = 'mlflow-artifacts'

def get_minio_client():
    """Get MinIO/S3 client"""
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

def load_data():
    """Load sales data"""
    return pd.read_parquet(DATA_PATH)

def prepare_features(df, target='sales'):
    """Prepare features for training"""
    features = ['day_of_week', 'week_of_year', 'season', 'cost', 
                'gross_sales', 'profit', 'rolling_mean_7', 'rolling_std_7', 
                'rolling_max_7', 'quantity', 'model', 'price_band']
    X = df[features]
    y = df[target]

    return X, y

def save_model_to_minio(model, model_name, model_type='sales'):
    """Save model to MinIO isntead of local storage"""
    s3 = get_minio_client()

    # Ensure bucket exists
    try:
        s3.head_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket {BUCKET_NAME} exists.")
    except:
        s3.create_bucket(Bucket=BUCKET_NAME)
        print(f"Bucket {BUCKET_NAME} created.")

    # Create model path
    path = f"{model_type}_prediction/models/"

    # Save model to bytes
    if model_name == 'Catboost':
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
            model.save_model(tmp.name)
            with open(tmp.name, 'rb') as f:
                model_bytes = f.read()
            os.unlink(tmp.name)
        key = f"{path}catboost.cbm"
    else:
        model_bytes = io.BytesIO()
        joblib.dump(model, model_bytes)
        model_bytes.seek(0)
        key = f"{path}{model_name.lower().replace(' ', '_')}.pkl"

    # Uplaod to MinIO
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=model_bytes if model_name == 'Catboost' else model_bytes.getvalue()
    )

    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'minio_bucket': BUCKET_NAME,
        'minio_path': key
    }

    metadata_key = f"{path}{model_name.lower().replace(' ', '_')}_metadata.json"
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=metadata_key,
        Body=json.dumps(metadata, indent=2)
    )

    print(f"✅ Saved {model_name} to MinIO: {key}")
    return metadata

def load_model_from_minio(model_name, model_type='sales'):
    """Load model from MinIO"""
    s3 = get_minio_client()
    path = f"{model_type}_prediction/models/"

    if model_name == 'Catboost':
        key = f"{path}catboost.cbm"
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
            tmp.write(response['Body'].read())
            tmp_path = tmp.name
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(tmp_path)
        os.unlink(tmp_path)
    else:
        key = f"{path}{model_name.lower().replace(' ', '_')}.pkl"
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        model_bytes = io.BytesIO(response['Body'].read())
        model = joblib.load(model_bytes)

    return model

def save_metrics_to_minio(metrics_df, model_type='sales'):
    """Save training metrics to MinIO"""
    s3 = get_minio_client()
    
    key = f"{model_type}_prediction/metrics/training_results.csv"
    csv_buffer = io.StringIO()
    metrics_df.to_csv(csv_buffer, index=False)
    
    s3.put_object(
        Bucket=BUCKET_NAME,
        Key=key,
        Body=csv_buffer.getvalue()
    )
    
    print(f"✅ Saved metrics to MinIO: {key}")

def load_metrics_from_minio(model_type='sales'):
    """Load metrics from MinIO"""
    s3 = get_minio_client()
    
    key = f"{model_type}_prediction/metrics/training_results.csv"
    try:
        response = s3.get_object(Bucket=BUCKET_NAME, Key=key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        return df
    except:
        return None

def list_models_from_minio():
    """List all models in MinIO"""
    s3 = get_minio_client()

    models = []
    for prefir in ['sales_prediction/models/', 'quantity_prediction/models/']:
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefir)
        for obj in response.get('Contents', []):
            if obj['Key'].endswith('.pkl', '.cbm'):
                models.append({
                    'path': obj['Key'],
                    'size': obj['Size'],
                    'last_modified': obj['LastModified'].isoformat()
                })

    return models