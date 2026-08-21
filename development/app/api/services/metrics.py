"""
Minimal Prometheus metrics for FastAPI
"""

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    REGISTRY,
    CollectorRegistry,
    multiprocess,
    ProcessCollector,
    GCCollector,
    PlatformCollector,
)
import pandas as pd
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from pathlib import Path
import time
import os
import logging
from functools import wraps
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define base directory
BASE_DIR = Path(__file__).resolve().parents[3]
ENV_DIR = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_DIR)

# ==========================================
# Request metrics
# =========================================
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

# ==========================================
# Prediction metrics
# ==========================================
PREDICTION_COUNT = Counter("predictions_total", "Total predictions", ["model_type", "model_name"])

PREDICTION_LATENCY = Histogram(
    "prediction_duration_seconds",
    "Prediction latency",
    ["model_type"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0),
)

PREDICTION_VALUE = Gauge("prediction_values", "Prediction values", ["model_type", "model_name"])

PREDICTION_DISTRIBUTION = Histogram(
    "prediction_distribution",
    "Prediction distribution",
    ["model_type"],
    buckets=(0, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000),
)

# ==========================================
# Model performance metrics
# ==========================================
MODEL_ACCURACY = Gauge("model_r2_score", "Model R² score", ["model_type", "model_name"])

MODEL_VERSIONS = Gauge(
    "mlflow_model_versions_total", "Number of model versions in MLflow", ["model_name"]
)

MODEL_STAGES = Gauge("mlflow_model_stages", "Model by stage", ["model_name", "stage"])

# ============================================
# BUSINESS METRICS (Updated from dataset)
# ============================================
BUSINESS_REVENUE_IMPACT = Gauge("business_revenue_impact", "Total revenue impact in IDR")

BUSINESS_ACTIVE_USERS = Gauge(
    "business_active_users", "Total active users/customers engaging with car sales platform"
)

BUSINESS_DROPPED_IMPACT = Gauge("business_dropped_impact", "Total dropped revenue impact in IDR")

BUSINESS_QUANTITY_SOLD = Gauge(
    "business_quantity_sold", "Total quantity of cars sold = quantity / car_id"
)

BUSINESS_KPI_METRICS = Gauge(
    "business_kpi_metrics", "Key business KPI performance indicators", ["kpi_name"]
)

# ===========================================
# AIRFLOW METRICS
# ===========================================
AIRFLOW_DAG_STATUS = Gauge(
    "airflow_dag_status",
    "Airflow DAG status (1=running, 0=success, -1=failed)",
    ["dag_id", "task_id"],
)

AIRFLOW_DAG_DURATION = Histogram(
    "airflow_dag_duration_seconds",
    "Airflow DAG execution duration",
    ["dag_id"],
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600)
)

AIRFLOW_TASK_DURATION = Histogram(
    "airflow_task_duration_seconds",
    "Airflow task execution duration",
    ["dag_id", "task_id"],
    buckets=(5, 10, 30, 60, 120, 300, 600)
)

AIRFLOW_DAG_RUNS = Counter(
    "airflow_dag_runs_total",
    "Total Airflow DAG runs",
    ["dag_id", "status"],
)

AIRFLOW_TASKS = Counter(
    "airflow_tasks_total",
    "Total Airflow tasks executed",
    ["dag_id", "task_id", "status"],
)

AIRFLOW_SCHEDULER_HEARTBEAT = Gauge(
    "airflow_scheduler_heartbeat",
    "Airflow scheduler heartbeat timestamp",
)

# ===========================================
# MIDDLEWARE CLASS
# ===========================================
class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically track HTTP request counts and latencies"""

    async def dispatch(self, request: Request, call_next):
        endpoint = request.url.path

        """Important to avoid unnecessary metrics noise for Prometheus scraping"""
        EXCLUDE_PATHS = {"/metrics", "/docs", "/openapi.json", "/redoc"}

        if endpoint in EXCLUDE_PATHS:
            return await call_next(request)

        method = request.method
        start_time = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception:
            status_code = 500
            raise

        finally:
            duration = time.perf_counter() - start_time
            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

# ============================================
# HELPER FUNCTIONS
# ============================================
def update_business_metrics():
    """Update business metrics from dataset"""
    try:
        # Find parquet file
        base_dir = Path(__file__).resolve().parents[3]
        parquet_path = base_dir / "database" / "car_sales_prediction_sales.parquet"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)

            # Integrate metrics to Dataframe to enchance business insights
            BUSINESS_REVENUE_IMPACT.set(float(df["sales"].sum()) if "sales" in df.columns else 0)
            BUSINESS_ACTIVE_USERS.set(float(df["customer_name"].nunique()) if "customer_name" in df.columns else 0)

            # Update KPIs
            if "sales" in df.columns:
                BUSINESS_KPI_METRICS.labels(kpi_name="total_sales").set(float(df["sales"].sum()))
            if "profit" in df.columns:
                BUSINESS_KPI_METRICS.labels(kpi_name="total_profit").set(float(df["profit"].sum()))
            if "quantity" in df.columns:
                BUSINESS_KPI_METRICS.labels(kpi_name="total_quantity").set(float(df["quantity"].sum()))
            if "profit_margin" in df.columns:
                BUSINESS_KPI_METRICS.labels(kpi_name="profit_margin").set(float(df["profit_margin"].mean()))

            return True

    except Exception as e:
        logger.error(f"❌ Error updating business metrics: {e}")

    return False

def update_airflow_metrics():
    """Update Airflow metrics from database using psycopg (v3)"""
    import os
    import time
    import psycopg

    try:
        # Construct connection string or pass kwargs directly
        conn_info = (
            f"host={os.getenv('POSTGRES_AIRFLOW_HOST')} "
            f"dbname={os.getenv('POSTGRES_AIRFLOW_DB')} "
            f"user={os.getenv('POSTGRES_AIRFLOW_USER')} "
            f"password={os.getenv('POSTGRES_AIRFLOW_PASSWORD')}"
        )

        # Context manager automatically handles closing connections and cursors ⚡
        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cur:
                # Get DAG status
                cur.execute("""
                    SELECT dag_id, state, COUNT(*)
                    FROM dag_run
                    WHERE state IN ('running', 'success', 'failed')
                    GROUP BY dag_id, state
                """)
                for dag_id, state, count in cur.fetchall():
                    status_value = 1 if state == 'running' else (0 if state == 'success' else -1)
                    AIRFLOW_DAG_STATUS.labels(dag_id=dag_id, task_id='all').set(status_value)
                    AIRFLOW_DAG_RUNS.labels(dag_id=dag_id, status=state).inc(count)

                # Get task status
                cur.execute("""
                    SELECT dag_id, task_id, state, COUNT(*)
                    FROM task_instance
                    WHERE state IN ('success', 'failed')
                    GROUP BY dag_id, task_id, state
                """)
                for dag_id, task_id, state, count in cur.fetchall():
                    AIRFLOW_TASKS.labels(
                        dag_id=dag_id,
                        task_id=task_id,
                        status=state
                    ).inc(count)

                # Update scheduler heartbeat
                cur.execute("SELECT NOW()")
                AIRFLOW_SCHEDULER_HEARTBEAT.set(time.time())

    except Exception as e:
        logger.error(f"❌ Error updating Airflow metrics: {e}")

def track_request(endpoint):
    """Decorator to track request metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            method = "POST" if hasattr(func, "post") else "GET"

            try:
                result = await func(*args, **kwargs)
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=200).inc()
                return result

            except Exception:
                REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=500).inc()
                raise

            finally:
                REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(time.perf_counter() - start)

        return wrapper
    return decorator

def track_prediction(model_type, model_name):
    """Decorator to track prediction metrics"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()

            result = await func(*args, **kwargs)

            PREDICTION_COUNT.labels(model_type=model_type, model_name=model_name).inc()
            PREDICTION_LATENCY.labels(model_type=model_type).observe(time.perf_counter() - start)

            # Extract value whether result is a Pydantic model or a dict
            pred_val = None
            if hasattr(result, "prediction"):
                pred_val = result.prediction
            elif isinstance(result, dict) and "prediction" in result:
                pred_val = result["prediction"]

            if pred_val is not None:
                PREDICTION_VALUE.labels(model_type=model_type, model_name=model_name).set(
                    float(pred_val)
                )

            return result
        return wrapper
    return decorator

def get_metrics():
    """Generates Prometheus metrics output for multiprocess/single-process setups."""
    # Update metrics before returning
    update_business_metrics()
    update_airflow_metrics()

    registry = CollectorRegistry()

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:

        # Collect custom worker & attachment processing metrics
        multiprocess.MultiProcessCollector(registry)
        ProcessCollector(registry=registry)
        GCCollector(registry=registry)
        PlatformCollector(registry=registry)
    else:
        registry = REGISTRY

    return generate_latest(registry)

def get_business_metrics(format_type="json"):
    """Return business metrics insight in requested format"""
    data = {
        "business_revenue_impact": BUSINESS_REVENUE_IMPACT._value.get(),
        "business_active_users": BUSINESS_ACTIVE_USERS._value.get(),
        "business_dropped_impact": BUSINESS_DROPPED_IMPACT._value.get(),
        "business_quantity_sold": BUSINESS_QUANTITY_SOLD._value.get(),
        "business_kpi": {},
    }

    if format_type == "json":
        return data
    return get_metrics()
