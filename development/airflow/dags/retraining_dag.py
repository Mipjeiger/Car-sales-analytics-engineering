from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import logging

"""
Airflow DAG for Model Retraining - Manual Trigger Only
Triggers both ML and CV training when run
"""

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]

default_args = {
    "owner": "ml_engineer",
    "depends_on_past": False,
    "start_date": datetime(2022, 1, 1),
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "model_retraining_pipeline",  # DAG ID
    default_args=default_args,
    description="Retrain all models - Manual trigger only",
    schedule=None,  # No automatic schedule
    catchup=False,
    tags=["ml", "retraining"],
)


def check_data_changes():
    """Check data status before retraining"""
    data_path = BASE_DIR / "development" / "database" / "car_prediction_sales.parquet"

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found at {data_path}")

    last_modified = data_path.stat().st_mtime
    logger.info(f"📊 Data last modified: {datetime.fromtimestamp(last_modified)}")

    # Read data stats
    df = pd.read_parquet(data_path)
    logger.info(f"📈 Data size: {len(df)} records")
    logger.info(f"📊 Data columns: {list(df.columns)}")

    return True


def check_models_exist():
    """Check if models exist in MinIO"""
    from training.utils import list_models_from_minio

    models = list_models_from_minio()
    if models:
        logger.info(f"✅ Models found in MinIO: {len(models)}")
        for m in models[:3]:  # Show first 3
            logger.info(f"   - {m['path']} ({m['size']} bytes)")
    else:
        logger.warning("⚠️ No models found in MinIO")

    return True


# Tasks
t0 = PythonOperator(
    task_id="check_data_changes",
    python_callable=check_data_changes,
    dag=dag,
)

t1 = PythonOperator(
    task_id="check_models_exist",
    python_callable=check_models_exist,
    dag=dag,
)

# Trigger ML training
t2 = TriggerDagRunOperator(
    task_id="trigger_ml_training",
    trigger_dag_id="ml_training_pipeline",
    wait_for_completion=True,
    dag=dag,
)

# Trigger computer vision training (runs after ML training completes)
t3 = TriggerDagRunOperator(
    task_id="trigger_cv_training",
    trigger_dag_id="cv_training_pipeline",
    wait_for_completion=True,
    dag=dag,
)


def send_completion_notification():
    """Send notification that retraining is completed"""
    logger.info("✅ Model retraining pipeline completed successfully.")
    logger.info("📊 Models updated in MinIO and registered in MLflow")
    return True


t4 = PythonOperator(
    task_id="send_completion_notification",
    python_callable=send_completion_notification,
    dag=dag,
)

# Define task dependencies
t0 >> t1 >> t2 >> t3 >> t4
