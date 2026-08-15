from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import sys
from pathlib import Path
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from training.ml_pipeline import MLTrainer

# Setup default arguments for the DAG
default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2022, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'ml_training_pipeline',  # DAG ID
    default_args=default_args,
    description='Train ML models for car sales & quantity prediction - Manual trigger only',
    schedule=None,  # No automatic schedule
    catchup=False,
    tags=['ml', 'training'],
)

def run_ml_training():
    """Run ML training"""
    trainer = MLTrainer()
    trainer.run()

def validate_models_in_minio():
    """Validate models exists in MinIO"""
    from training.utils import list_models_from_minio

    models = list_models_from_minio()
    if not models:
        raise ValueError("No models found in MinIO after training.")

    logger.info(f"✅ Models validated in MinIO: {models}")
    for m in models[:5]: # Log first 5 models
        logger.info(f"   - {m['path']} ({m['size']} bytes)")

def register_models_to_mlflow():
    """Load models from MinIO and register them to MLflow"""
    from mlflow.load_existing_models import MLflowModelLoader
    loader = MLflowModelLoader()
    loader.load_all()

# Tasks
t1 = PythonOperator(
    task_id='run_ml_training',
    python_callable=run_ml_training,
    dag=dag,
)

t2 = PythonOperator(
    task_id='validate_models_in_minio',
    python_callable=validate_models_in_minio,
    dag=dag,
)

t3 = PythonOperator(
    task_id='register_models_to_mlflow',
    python_callable=register_models_to_mlflow,
    dag=dag,
)

# Define task dependencies
t1 >> t2 >> t3