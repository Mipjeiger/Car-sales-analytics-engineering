"""
Evaluate models and save metrics for DVC with MinIO
"""

import json
from dotenv import load_dotenv
import pandas as pd
import mlflow
import os
from pathlib import Path


def evaluate_models():
    SCRIPT_DIR = Path(__file__).resolve().parent  # development/pipeline/stages
    DEV_DIR = SCRIPT_DIR.parents[1]  # development
    DVC_METRICS = DEV_DIR / "dvc" / "metrics"
    DVC_METRICS.mkdir(parents=True, exist_ok=True)

    # Load environment variables from development/.env
    env_path = DEV_DIR / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)

    # Setup MLflow
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"📡 Connected to MLflow Tracking Server at: {tracking_uri}")

    metrics = {"total_runs": 0}

    try:
        runs = mlflow.search_runs(search_all_experiments=True)

        if not runs.empty:
            r2_col = next(
                (
                    col
                    for col in ["metrics.R2", "metrics.r2_score", "metrics.r2"]
                    if col in runs.columns
                ),
                None,
            )
            rmse_col = next(
                (col for col in ["metrics.RMSE", "metrics.rmse"] if col in runs.columns), None
            )

            if r2_col and not runs[r2_col].isna().all():
                best_idx = runs[r2_col].idxmax()
                best = runs.loc[best_idx]

                run_name = best.get("tags.mlflow.runName") or best.get("run_id", "unknown")
                r2_val = float(best.get(r2_col, 0)) if pd.notna(best.get(r2_col)) else 0.0
                rmse_val = (
                    float(best.get(rmse_col, 0))
                    if (rmse_col and pd.notna(best.get(rmse_col)))
                    else 0.0
                )

                metrics = {
                    "best_model": {
                        "name": str(run_name),
                        "r2": r2_val,
                        "rmse": rmse_val,
                        "run_id": str(best.get("run_id", "unknown")),
                    },
                    "total_runs": len(runs),
                }

            else:
                metrics["total_runs"] = len(runs)

    except Exception as e:
        print(f"⚠️ Error querying MLflow runs: {e}")

    # Save metrics to evaluation.json
    output_file = DVC_METRICS / "evaluation.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ Metrics saved: {metrics['total_runs']} runs")


if __name__ == "__main__":
    evaluate_models()
