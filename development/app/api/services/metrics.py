"""
Minimal Prometheus metrics for FastAPI
"""

from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY

# Request metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'Request latency',
    ['method', 'endpoint'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0)
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    'predictions_total',
    'Total predictions',
    ['model_type', 'model_name']
)

PREDICTION_LATENCY = Histogram(
    'prediction_duration_seconds',
    'Prediction latency',
    ['model_type'],
    buckets=(0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0)
)

# Model accuracy
MODEL_ACCURACY = Gauge(
    'model_r2_score',
    'Model R² score',
    ['model_type', 'model_name']
)

# Business Impact metrics
BUSINESS_REVENUE_IMPACT = Gauge(
    'business_revenue_impact',
    'Estimated or cumulative revenue impact in IDR'
)

BUSINESS_ACTIVE_USERS = Gauge(
    'business_active_users',
    'Total active users/customers engaging with car sales platform'
)

BUSINESS_KPI_METRICS = Gauge(
    'business_kpi_metrics',
    'Key business KPI performance indicators',
    ['kpi_name']
)

def update_business_metrics():
    """Calculates and updates business metrics gauges from dataset or default values."""
    try:
        import pandas as pd
        from pathlib import Path
        
        candidate_paths = [
            Path(__file__).resolve().parents[3] / "development" / "database" / "car_sales_prediction_sales.parquet",
            Path(__file__).resolve().parents[2] / "database" / "car_sales_prediction_sales.parquet",
            Path("/app/development/database/car_sales_prediction_sales.parquet"),
            Path("/app/database/car_sales_prediction_sales.parquet"),
        ]
        parquet_path = next((p for p in candidate_paths if p.exists()), None)
        
        if parquet_path:
            df = pd.read_parquet(parquet_path)
            total_sales = float(df['sales'].sum()) if 'sales' in df.columns else 24265599235630.0
            active_users = float(df['customer_name'].nunique()) if 'customer_name' in df.columns else 3021.0
            total_profit = float(df['profit'].sum()) if 'profit' in df.columns else 3123796070787.0
            total_qty = float(df['quantity'].sum()) if 'quantity' in df.columns else 53963.0
            avg_margin = float(df['profit_margin'].mean()) if 'profit_margin' in df.columns else 10.17
        else:
            total_sales = 24265599235630.0
            active_users = 3021.0
            total_profit = 3123796070787.0
            total_qty = 53963.0
            avg_margin = 10.17

        BUSINESS_REVENUE_IMPACT.set(total_sales)
        BUSINESS_ACTIVE_USERS.set(active_users)
        BUSINESS_KPI_METRICS.labels(kpi_name="total_sales").set(total_sales)
        BUSINESS_KPI_METRICS.labels(kpi_name="total_profit").set(total_profit)
        BUSINESS_KPI_METRICS.labels(kpi_name="total_quantity").set(total_qty)
        BUSINESS_KPI_METRICS.labels(kpi_name="profit_margin").set(avg_margin)
        
        return {
            "business_revenue_impact": total_sales,
            "business_active_users": active_users,
            "business_kpis": {
                "total_sales": total_sales,
                "total_profit": total_profit,
                "total_quantity": total_qty,
                "profit_margin": avg_margin
            }
        }
    except Exception:
        BUSINESS_REVENUE_IMPACT.set(0)
        BUSINESS_ACTIVE_USERS.set(0)
        return {
            "business_revenue_impact": 0,
            "business_active_users": 0,
            "business_kpis": {}
        }

def get_metrics():
    """Return Prometheus metrics"""
    return generate_latest(REGISTRY)

def get_business_metrics(format_type: str = "prometheus"):
    """Update business metrics gauges and return metrics in requested format."""
    data = update_business_metrics()
    if format_type == "json":
        return data
    return generate_latest(REGISTRY)