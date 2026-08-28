from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import logging
from pathlib import Path
from training.cv_training import CVTrainer

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration directory and path setup
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

default_args = {
    "owner": "cv_engineer",
    "depends_on_past": False,
    "start_date": datetime(2022, 1, 1),
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "cv_training_pipeline",  # DAG ID
    default_args=default_args,
    description="Train CV models for car sales prediction - Manual trigger only",
    schedule=None,  # No automatic schedule
    catchup=False,
    is_paused_upon_creation=False,
    tags=["cv", "training", "car_sales"],
)

def run_cv_training():
    """Run CV training"""
    logger.info("🚀 Starting CV training...")
    trainer = CVTrainer()
    trainer.train()

def validate_cv_model():
    """Validate CV model in MinIO"""
    import boto3
    from botocore.client import Config
    import os

    s3 = boto3.client(
        "s3",
        endpoint_url=os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )
    
    try:
        s3.head_object(Bucket="mlflow-artifacts", Key="computer_vision/car_index.faiss")
        logger.info("✅ CV model validated in MinIO")
    except:
        raise ValueError("❌ CV model not found in MinIO")

def register_cv_mlflow():
    """Load CV model from MinIO to MLflow"""
    from mlflow_utils.load_existing_models import MLflowModelLoader
    loader = MLflowModelLoader()
    loader.load_cv_models()

t1 = PythonOperator(
    task_id="run_cv_training",
    python_callable=run_cv_training,
    dag=dag,
)

t2 = PythonOperator(
    task_id="validate_cv_model",
    python_callable=validate_cv_model,
    dag=dag,
)

t3 = PythonOperator(
    task_id="register_cv_mlflow",
    python_callable=register_cv_mlflow,
    dag=dag,
)

# Define task dependencies
t1 >> t2 >> t3