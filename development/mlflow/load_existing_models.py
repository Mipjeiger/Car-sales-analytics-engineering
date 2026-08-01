"""
Load existing models from development/models into MLflow
"""

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class MLflowModelLoader:
    def __init__(self):
        # Resolve path dynamically relative to this script
        self.SCRIPT_DIR = Path(__file__).resolve().parent
        self.BASE_DIR = self.SCRIPT_DIR.parent
        self.MODELS_DIR = self.BASE_DIR / "models"
        self.MLFLOW_DIR = self.BASE_DIR / "mlflow"

        self.MLFLOW_DIR.mkdir(exist_ok=True)

        # Smart URI fallback: Uses MLFLOW_TRACKING_URI or defaults to localhost if run outside Docker
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"📡 Connected to MLflow Tracking Server at: {tracking_uri}")

    def load_sales_models(self):
        """Load all sales prediction models"""
        sales_dir = self.MODELS_DIR / "sales_prediction"
        metrics_path = sales_dir / "metrics" / "model_metrics.csv"

        if not metrics_path.exists():
            print(f"⚠️ Metrics file not found: {metrics_path}")
            return

        metrics_df = pd.read_csv(metrics_path)
        model_files = {
            "XGBoost": "xgboost.pkl",
            "Random Forest": "random_forest.pkl",
            "Decision Tree": "decision_tree.pkl",
            "CatBoost": "catboost.cbm",
        }

        print("\n📈 [1/3] Logging Sales Models...")
        for _, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc="Sales Models"):
            model_name = row["Model"]
            model_file = model_files.get(model_name)
            if not model_file:
                continue

            model_path = sales_dir / "models" / model_file
            if not model_path.exists():
                print(f"\n⚠️ {model_name} not found: {model_path}")
                continue

            try:
                with mlflow.start_run(run_name=f"sales_{model_name}"):
                    if model_name == "CatBoost":
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor()
                        model.load_model(model_path)
                    else:
                        model = joblib.load(model_path)

                    mlflow.sklearn.log_model(model, f"sales_{model_name}")
                    mlflow.log_metrics({
                        "R2": float(row.get("R2", 0)),
                        "RMSE": float(row.get("RMSE", 0)),
                        "MAE": float(row.get("MAE", 0)),
                    })

                    params_file = sales_dir / "parameters" / f"{model_name.lower().replace(' ', '_')}_best_params.json"
                    if params_file.exists():
                        with open(params_file, "r") as f:
                            mlflow.log_params(json.load(f))

                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/sales_{model_name}",
                        f"SalesPredictor_{model_name}",
                    )

            except Exception as e:
                print(f"\n❌ Failed to load {model_name}: {e}")

    def load_quantity_models(self):
        """Load all quantity prediction models"""
        qty_dir = self.MODELS_DIR / "quantity_prediction"
        metrics_path = qty_dir / "metrics" / "model_metrics.csv"

        if not metrics_path.exists():
            print(f"⚠️ Metrics file not found: {metrics_path}")
            return

        metrics_df = pd.read_csv(metrics_path)
        model_files = {
            "XGBoost": "xgboost.pkl",
            "Random Forest": "random_forest.pkl",
            "Decision Tree": "decision_tree.pkl",
            "CatBoost": "catboost.cbm",
        }

        print("\n📦 [2/3] Logging Quantity Models...")
        for _, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc="Quantity Models"):
            model_name = row["Model"]
            model_file = model_files.get(model_name)
            if not model_file:
                continue

            model_path = qty_dir / "models" / model_file
            if not model_path.exists():
                print(f"\n⚠️ {model_name} not found: {model_path}")
                continue

            try:
                with mlflow.start_run(run_name=f"quantity_{model_name}"):
                    if model_name == "CatBoost":
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor()
                        model.load_model(model_path)
                    else:
                        model = joblib.load(model_path)

                    mlflow.sklearn.log_model(model, f"quantity_{model_name}")
                    mlflow.log_metrics({
                        "R2": float(row.get("R2", 0)),
                        "RMSE": float(row.get("RMSE", 0)),
                        "MAE": float(row.get("MAE", 0)),
                    })

                    params_file = qty_dir / "parameters" / f"{model_name.lower().replace(' ', '_')}_best_params.json"
                    if params_file.exists():
                        with open(params_file, "r") as f:
                            mlflow.log_params(json.load(f))

                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/quantity_{model_name}",
                        f"QuantityPredictor_{model_name}",
                    )

            except Exception as e:
                print(f"\n❌ Failed to load {model_name}: {e}")

    def load_cv_models(self):
        """Load computer vision models"""
        cv_dir = self.MODELS_DIR / "computer_vision_2"
        if not cv_dir.exists():
            cv_dir = self.MODELS_DIR / "computer_vision"

        if not cv_dir.exists():
            print("⚠️ Computer vision directory not found")
            return

        print("\n👁️ [3/3] Logging Computer Vision Artifacts...")
        try:
            with mlflow.start_run(run_name="computer_vision_faiss"):
                index_path = cv_dir / "car_index.faiss"
                if index_path.exists():
                    mlflow.log_artifact(str(index_path))

                metadata_path = cv_dir / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                        mlflow.log_params({
                            "num_images": metadata.get("num_images", 0),
                            "feature_dimension": metadata.get("feature_dimension", 0),
                        })

                for file in ["feature_data.csv", "feature_matrix.npy", "brand_mapping.json"]:
                    file_path = cv_dir / file
                    if file_path.exists():
                        mlflow.log_artifact(str(file_path))

                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}",
                    "ComputerVision_FAISS",
                )
                print("✅ Computer Vision model registered!")

        except Exception as e:
            print(f"❌ Failed to load CV models: {e}")

    def load_all(self):
        """Load all models to MLflow"""
        print("🚀 Starting MLflow Loader...")
        print("=" * 50)

        self.load_sales_models()
        self.load_quantity_models()
        self.load_cv_models()

        print("\n" + "=" * 50)
        print("✅ All models successfully loaded to MLflow!")
        print("📊 MLflow UI: http://localhost:5003")


if __name__ == "__main__":
    loader = MLflowModelLoader()
    loader.load_all()