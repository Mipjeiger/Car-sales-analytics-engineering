"""
Evaluate models and save metrics for DVC with MinIO
"""

import json
import pandas as pd
import mlflow
import boto3
from pathlib import Path
from botocore.client import Config

def evaluate_models():
    BASE_DIR = Path.cwd()
    DVC_METRICS = BASE_DIR / 'development' / 'dvc' / 'metrics'
    DVC_METRICS.mkdir(parents=True, exist_ok=True)
    
    # Setup MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    experiment = mlflow.get_experiment_by_name('car_sales_training')
    metrics = {'total_runs': 0}
    
    if experiment:
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        if not runs.empty:
            best_idx = runs['metrics.r2_score'].idxmax()
            best = runs.loc[best_idx]
            
            metrics = {
                'best_model': {
                    'name': best.get('tags.mlflow.runName', 'Unknown'),
                    'r2': float(best.get('metrics.r2_score', 0)),
                    'rmse': float(best.get('metrics.rmse', 0)),
                    'run_id': best.get('run_id', '')
                },
                'total_runs': len(runs)
            }
    
    # Save metrics
    with open(DVC_METRICS / 'evaluation.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"✅ Metrics saved: {metrics['total_runs']} runs")

if __name__ == "__main__":
    evaluate_models()