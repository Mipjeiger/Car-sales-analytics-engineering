from botocore.client import Config
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import boto3
import sys
import os
import traceback
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set HF TOKEN environment variable for Hugging Face model access
# Fix: Use empty string as fallback instead of None
hf_token = os.getenv("HUGGINGFACE_API_KEY")
if hf_token is not None:
    os.environ["HF_TOKEN"] = hf_token
else:
    os.environ["HF_TOKEN"] = "" 
    logger.warning("⚠️ HUGGINGFACE_API_KEY not set, using empty string")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Use the mounted /app path for project root
PROJECT_ROOT = Path("/app")

# Fallback if /app doesn't exist
if not PROJECT_ROOT.exists():
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    logger.warning(f"⚠️ /app not found, using: {PROJECT_ROOT}")

logger.info(f"📁 Project root: {PROJECT_ROOT}")

# Add project root to Python path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "development"))

default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2022, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# Define the DAG - Fix: use schedule_interval instead of schedule
dag = DAG(
    'cars_damage_identification_pipeline',
    default_args=default_args,
    description='Train and validate cars damage identification model - Manual trigger only',
    schedule_interval=None,  # Changed from schedule=None to schedule_interval=None
    catchup=False,
    is_paused_upon_creation=False,
    tags=['cv', 'training', 'car_damage'],
)

def run_damage_training():
    """Run car damage identification training"""
    try:
        # Set environment variables for HuggingFace - handle None values
        hf_token = os.getenv("HUGGINGFACE_API_KEY")
        if hf_token is not None:
            os.environ["HF_TOKEN"] = hf_token
        else:
            os.environ["HF_TOKEN"] = ""
        
        os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
        
        # Ensure Python path has the training module
        if str(PROJECT_ROOT) not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT))
        if str(PROJECT_ROOT / "development") not in sys.path:
            sys.path.insert(0, str(PROJECT_ROOT / "development"))

        # Try to import and run training
        from training.cars_damage_identification import run_training
        logger.info("🚀 Starting cars damage identification training...")    

        # Run training
        trainer, search, history = run_training()

        if trainer is None:
            raise ValueError("❌ Training failed. Trainer is None.")

        logger.info("✅ Training completed successfully!")
        if history:
            logger.info(f"📊 Training history: {history[-3:]}")
        else:
            logger.info("📊 No training history available")
        
        if search and hasattr(search, 'index') and search.index is not None:
            logger.info(f"🔍 Searcher index built with {search.index.ntotal} entries.")
        
        return True

    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        training_path = PROJECT_ROOT / "development" / "training"
        if training_path.exists():
            logger.info(f"📁 Training directory exists at: {training_path}")
            logger.info(f"📁 Contents: {[f.name for f in training_path.iterdir()]}")
        else:
            logger.error(f"❌ Training directory not found at: {training_path}")
        raise
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

def verify_model_saved():
    """Verify that the model was saved correctly after training"""
    model_paths = [
        PROJECT_ROOT / "models" / "quality_assurance",
        Path("/app/models/quality_assurance"),
        Path("/opt/airflow/models/quality_assurance"),
    ]
    
    model_found = False
    for model_dir in model_paths:
        if model_dir.exists():
            logger.info(f"🔍 Checking model directory: {model_dir}")
            files = list(model_dir.iterdir())
            logger.info(f"📁 Local model files: {[f.name for f in files]}")
            model_found = True
            break
    
    if not model_found:
        logger.warning("⚠️ No model directory found")

    # Check MinIO if configured
    try:
        # Handle None values for MinIO credentials
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if access_key is None or secret_key is None:
            logger.warning("⚠️ MinIO credentials not set, skipping MinIO check")
            return
        
        s3 = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000'),
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        
        bucket = os.getenv('MLFLOW_ARTIFACT_BUCKET', 'mlflow-artifacts')
        try:
            response = s3.list_objects_v2(Bucket=bucket, Prefix='quality_assurance/')
            if 'Contents' in response:
                files = [obj['Key'] for obj in response['Contents']]
                logger.info(f"📁 MinIO model files: {files}")
        except Exception as e:
            logger.error(f"❌ Error accessing MinIO: {e}")
    except Exception as e:
        logger.warning(f"⚠️ MinIO client error: {e}")

# Create tasks
train_task = PythonOperator(
    task_id='run_damage_training',
    python_callable=run_damage_training,
    dag=dag,
)

verify_task = PythonOperator(
    task_id='verify_model_saved',
    python_callable=verify_model_saved,
    dag=dag,
)

# Set task dependencies
train_task >> verify_task