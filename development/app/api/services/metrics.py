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

def get_metrics():
    """Return Prometheus metrics"""
    return generate_latest(REGISTRY)