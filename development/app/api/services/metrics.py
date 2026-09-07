"""
Prometheus & Business Metrics Collector for FastAPI
"""

import logging
import os
import time
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Request
import pandas as pd
from prometheus_client import (
    REGISTRY,
    CollectorRegistry,
    Counter,
    GCCollector,
    Gauge,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
    multiprocess,
)

# Optional dependencies for systemic metrics (graceful fallback)
try:
    import psutil
except ImportError:
    psutil = None

try:
    import psycopg
except ImportError:
    psycopg = None

from starlette.middleware.base import BaseHTTPMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Define base directory
BASE_DIR = Path(__file__).resolve().parents[3]
ENV_DIR = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_DIR)

# App Uptime Reference
APP_START_TIME = time.time()

# ==========================================
# 1. HTTP & WEBSITE TRAFFIC METRICS 🌐
# ==========================================
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0),
)

HTTP_ACTIVE_REQUESTS = Gauge(
    "http_active_requests", "Number of active concurrent HTTP requests"
)

HTTP_ERRORS_TOTAL = Counter(
    "http_errors_total", "Total count of HTTP error responses", ["status_code", "endpoint"]
)

# ==========================================
# 2. SYSTEM & HOST RESOURCE METRICS 💻
# ==========================================
SYSTEM_CPU_USAGE = Gauge("system_cpu_usage_percent", "Current CPU utilization percentage")
SYSTEM_MEMORY_USAGE = Gauge("system_memory_usage_bytes", "Current RAM memory usage in bytes")
SYSTEM_MEMORY_PERCENT = Gauge("system_memory_usage_percent", "Current RAM memory utilization percentage")
SYSTEM_UPTIME_SECONDS = Gauge("system_uptime_seconds", "Total runtime of the API service in seconds")

# ==========================================
# 3. DATABASE (POSTGRESQL) HEALTH METRICS 🗄️
# ==========================================
DB_CONNECTIVITY_STATUS = Gauge(
    "db_connection_status", "PostgreSQL Connection status (1=connected, 0=disconnected)", ["db_name"]
)
DB_ACTIVE_CONNECTIONS = Gauge(
    "db_active_connections", "Number of active PostgreSQL connections", ["db_name"]
)

# ==========================================
# 4. PREDICTION METRICS 🤖
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
# 5. MODEL PERFORMANCE METRICS 📈
# ==========================================
MODEL_ACCURACY = Gauge("model_r2_score", "Model R² score", ["model_type", "model_name"])

MODEL_VERSIONS = Gauge(
    "mlflow_model_versions_total", "Number of model versions in MLflow", ["model_name"]
)

MODEL_STAGES = Gauge("mlflow_model_stages", "Model by stage", ["model_name", "stage"])

# ==========================================
# 6. BUSINESS METRICS 📊
# ==========================================
BUSINESS_REVENUE_IMPACT = Gauge("business_revenue_impact", "Total revenue impact in IDR")
BUSINESS_ACTIVE_USERS = Gauge("business_active_users", "Total active users/customers engaging with platform")
BUSINESS_DROPPED_IMPACT = Gauge("business_dropped_impact", "Total dropped revenue impact in IDR")
BUSINESS_QUANTITY_SOLD = Gauge("business_quantity_sold", "Total quantity of cars sold")
BUSINESS_KPI_METRICS = Gauge("business_kpi_metrics", "Key business KPI performance indicators", ["kpi_name"])

# ==========================================
# 7. AIRFLOW PIPELINE METRICS ⚡
# ==========================================
AIRFLOW_DAG_STATUS = Gauge(
    "airflow_dag_status", "Airflow DAG status (1=running, 0=success, -1=failed)", ["dag_id", "task_id"]
)
AIRFLOW_DAG_DURATION = Histogram(
    "airflow_dag_duration_seconds", "Airflow DAG execution duration", ["dag_id"],
    buckets=(10, 30, 60, 120, 300, 600, 1800, 3600)
)
AIRFLOW_TASK_DURATION = Histogram(
    "airflow_task_duration_seconds", "Airflow task execution duration", ["dag_id", "task_id"],
    buckets=(5, 10, 30, 60, 120, 300, 600)
)
AIRFLOW_DAG_RUNS = Counter("airflow_dag_runs_total", "Total Airflow DAG runs", ["dag_id", "status"])
AIRFLOW_TASKS = Counter("airflow_tasks_total", "Total Airflow tasks executed", ["dag_id", "task_id", "status"])
AIRFLOW_SCHEDULER_HEARTBEAT = Gauge("airflow_scheduler_heartbeat", "Airflow scheduler heartbeat timestamp")


# ==========================================
# ENHANCED MIDDLEWARE CLASS 🚀
# ==========================================
class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to automatically track HTTP request counts, active connections, and error metrics."""

    async def dispatch(self, request: Request, call_next):
        endpoint = request.url.path
        EXCLUDE_PATHS = {"/metrics", "/docs", "/openapi.json", "/redoc", "/favicon.ico"}

        if endpoint in EXCLUDE_PATHS:
            return await call_next(request)

        method = request.method
        start_time = time.perf_counter()
        status_code = 500

        # Increment active requests counter
        HTTP_ACTIVE_REQUESTS.inc()

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response

        except Exception:
            status_code = 500
            raise

        finally:
            duration = time.perf_counter() - start_time
            HTTP_ACTIVE_REQUESTS.dec()

            REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=status_code).inc()
            REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)

            # Track 4xx and 5xx errors specifically
            if status_code >= 400:
                HTTP_ERRORS_TOTAL.labels(status_code=str(status_code), endpoint=endpoint).inc()

# ==========================================
# METRIC UPDATER FUNCTIONS 🛠️
# ==========================================
def update_system_metrics():
    """Collect system resource usage metrics (CPU, RAM, Uptime)"""
    SYSTEM_UPTIME_SECONDS.set(time.time() - APP_START_TIME)

    if psutil:
        try:
            SYSTEM_CPU_USAGE.set(psutil.cpu_percent(interval=None))
            mem = psutil.virtual_memory()
            SYSTEM_MEMORY_USAGE.set(mem.used)
            SYSTEM_MEMORY_PERCENT.set(mem.percent)
        except Exception as e:
            logger.error(f"❌ Error updating system metrics: {e}")


def update_database_metrics():
    """Ping application PostgreSQL instance to measure health & active connections"""
    if not psycopg:
        return

    db_host = os.getenv("POSTGRES_HOST")
    db_name = os.getenv("POSTGRES_DB")
    db_user = os.getenv("POSTGRES_USER")
    db_pass = os.getenv("POSTGRES_PASSWORD")

    conn_info = f"host={db_host} dbname={db_name} user={db_user} password={db_pass} connect_timeout=3"

    try:
        with psycopg.connect(conn_info) as conn:
            DB_CONNECTIVITY_STATUS.labels(db_name=db_name).set(1)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) FROM pg_stat_activity WHERE datname = %s AND state = 'active';",
                    (db_name,),
                )
                res = cur.fetchone()
                if res:
                    DB_ACTIVE_CONNECTIONS.labels(db_name=db_name).set(res[0])
    except Exception as e:
        DB_CONNECTIVITY_STATUS.labels(db_name=db_name).set(0)
        logger.error(f"❌ PostgreSQL Health Check Failed ({db_name}): {e}")

def update_business_metrics():
    """Update business metrics from parquet storage"""
    try:
        parquet_path = BASE_DIR / "database" / "car_sales_prediction_sales.parquet"

        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)

            BUSINESS_REVENUE_IMPACT.set(float(df["sales"].sum()) if "sales" in df.columns else 0)
            BUSINESS_ACTIVE_USERS.set(float(df["customer_name"].nunique()) if "customer_name" in df.columns else 0)

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
    """Update Airflow DAG and Task metrics"""
    if not psycopg:
        return

    try:
        conn_info = (
            f"host={os.getenv('POSTGRES_AIRFLOW_HOST')} "
            f"dbname={os.getenv('POSTGRES_AIRFLOW_DB')} "
            f"user={os.getenv('POSTGRES_AIRFLOW_USER')} "
            f"password={os.getenv('POSTGRES_AIRFLOW_PASSWORD')}"
        )

        with psycopg.connect(conn_info) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT dag_id, state, COUNT(*)
                    FROM dag_run
                    WHERE state IN ('running', 'success', 'failed')
                    GROUP BY dag_id, state
                """)
                for dag_id, state, count in cur.fetchall():
                    status_value = 1 if state == "running" else (0 if state == "success" else -1)
                    AIRFLOW_DAG_STATUS.labels(dag_id=dag_id, task_id="all").set(status_value)
                    AIRFLOW_DAG_RUNS.labels(dag_id=dag_id, status=state).inc(count)

                cur.execute("""
                    SELECT dag_id, task_id, state, COUNT(*)
                    FROM task_instance
                    WHERE state IN ('success', 'failed')
                    GROUP BY dag_id, task_id, state
                """)
                for dag_id, task_id, state, count in cur.fetchall():
                    AIRFLOW_TASKS.labels(dag_id=dag_id, task_id=task_id, status=state).inc(count)

                AIRFLOW_SCHEDULER_HEARTBEAT.set(time.time())

    except Exception as e:
        logger.error(f"❌ Error updating Airflow metrics: {e}")


# ==========================================
# DECORATORS 🎯
# ==========================================
def track_request(endpoint):
    """Decorator to track specific internal function latencies"""
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
    """Decorator to track ML prediction values and runtime"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = await func(*args, **kwargs)

            PREDICTION_COUNT.labels(model_type=model_type, model_name=model_name).inc()
            PREDICTION_LATENCY.labels(model_type=model_type).observe(time.perf_counter() - start)

            pred_val = None
            if hasattr(result, "prediction"):
                pred_val = result.prediction
            elif isinstance(result, dict) and "prediction" in result:
                pred_val = result["prediction"]

            if pred_val is not None:
                PREDICTION_VALUE.labels(model_type=model_type, model_name=model_name).set(float(pred_val))

            return result
        return wrapper
    return decorator


# ==========================================
# EXPORT METRICS ENDPOINT REGISTRY 📡
# ==========================================
def get_metrics():
    """Generates Prometheus metrics payload after triggering collectors."""
    update_system_metrics()
    update_database_metrics()
    update_business_metrics()
    update_airflow_metrics()

    registry = CollectorRegistry()

    if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
        multiprocess.MultiProcessCollector(registry)
        ProcessCollector(registry=registry)
        GCCollector(registry=registry)
        PlatformCollector(registry=registry)
    else:
        registry = REGISTRY

    return generate_latest(registry)


def get_business_metrics(format_type="json"):
    """Return key business metrics for JSON APIs or Prometheus scrapers"""
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