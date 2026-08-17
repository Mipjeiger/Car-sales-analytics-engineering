from airflow.plugins_manager import AirflowPlugin
from flask import Blueprint, Response
from prometheus_client import generate_latest, Counter, Gauge, Histogram
from airflow.models import DagRun, TaskInstance
from airflow.utils.state import State
from sqlalchemy import func
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==================================
# Airfow Metrics Plugin
# ==================================
DAG_RUNS = Counter(
    'airflow_dag_runs_total',
    'Total DAG runs',
    ['dag_id', 'state']
)

TASK_RUNS = Counter(
    'airflow_task_runs_total',
    'Total Task runs',
    ['dag_id', 'task_id', 'state']
)

DAG_DURATION = Histogram(
    'airflow_dag_duration_seconds',
    'DAG run duration',
    ['dag_id'],
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600)
)

TASK_DURATION = Histogram(
    'airflow_task_duration_seconds',
    'Task run duration',
    ['dag_id', 'task_id'],
    buckets=(5, 10, 30, 60, 120, 300, 600)
)

ACTIVE_DAGS = Gauge(
    'airflow_active_dags',
    'Number of active DAGs',
    ['state']
)

def collect_metrics():
    """Collect Airflow metrics from database"""
    from airflow import settings
    session = settings.Session()

    try:
        # Get DAG run statistics
        dag_stats = session.query(
            DagRun.dag_id,
            DagRun.state,
            func.count(DagRun.id).label('count'),
            func.avg(DagRun.end_date - DagRun.start_date).label('avg_duration')
        ).group_by(DagRun.dag_id, DagRun.state).all()

        for dag_id, state, count, avg_duration in dag_stats:
            DAG_RUNS.labels(dag_id=dag_id, state=state).inc(count)

            if avg_duration:
                DAG_DURATION.labels(dag_id=dag_id).observe(avg_duration.total_seconds())

        # Get Task run statistics
        task_stats = session.query(
            TaskInstance.dag_id,
            TaskInstance.task_id,
            TaskInstance.state,
            func.count(TaskInstance.id).label('count'),
            func.avg(TaskInstance.duration).label('avg_duration')
        ).group_by(TaskInstance.dag_id, TaskInstance.task_id, TaskInstance.state).all()

        for dag_id, task_id, state, count, avg_duration in task_stats:
            TASK_RUNS.labels(dag_id=dag_id, task_id=task_id, state=state).inc(count)

            if avg_duration:
                TASK_DURATION.labels(dag_id=dag_id, task_id=task_id).observe(avg_duration)

        # Get active DAGs count
        running = session.query(DagRun).filter(DagRun.state == State.RUNNING).count()
        failed = session.query(DagRun).filter(DagRun.state == State.FAILED).count()
        success = session.query(DagRun).filter(DagRun.state == State.SUCCESS).count()

        # Update ACTIVE_DAGS gauge
        ACTIVE_DAGS.labels(state='running').set(running)
        ACTIVE_DAGS.labels(state='failed').set(failed)
        ACTIVE_DAGS.labels(state='success').set(success)

    except Exception as e:
        logger.error(f"Error collecting Airflow metrics: {e}")
    finally:
        session.close()

# ==========================
# Flask Blueprint for Airflow Metrics
# ==========================
metrics_bp = Blueprint('metrics', __name__)

@metrics_bp.route('/metrics')
def metrics():
    """Prometheus metrics endpoint for Airflow"""
    collect_metrics()
    return Response(generate_latest(), mimetype='text/plain')

# =========================
# Airflow Plugin Definition
# =========================
class AirflowMetricsPlugin(AirflowPlugin):
    name = "airflow_metrics_plugin"
    flask_blueprints = [metrics_bp]