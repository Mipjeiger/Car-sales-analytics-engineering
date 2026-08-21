from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

default_args = {
    'owner': 'ml_engineer',
    'depends_on_past': False,
    'start_date': datetime(2022, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'cars_damage_identification_pipeline',
    default_args=default_args,
    description='Train and validate cars damage identification model - Manual trigger only',
    schedule=None,  # No automatic schedule
    catchup=False,
    is_paused_upon_creation=False,
    tags=['cv', 'training', 'car_damage'],
)

def run_damage_training():
    """Run car damage identification training"""
    from training.cars_damage_identification import run_training
    trainer, search, history = run_training()

    if trainer is None:
        raise ValueError("❌ Training failed. Trainer is None.")

    logger.info("✅ Training completed successfully.")
    return True