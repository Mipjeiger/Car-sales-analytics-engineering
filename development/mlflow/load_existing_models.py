"""
Load existing models from development/models into MLflow using .env configuration
"""

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv


class MLflowModelLoader:
    def __init__(self):
        # Resolve directory hierarchy dynamically relative to this script
        # Script location: development/mlflow/load_existing_models.py
        self.SCRIPT_DIR = Path(__file__).resolve().parent  # development/mlflow
        self.DEV_DIR = self.SCRIPT_DIR.parent  # development
        self.BASE_DIR = self.DEV_DIR.parent  # project root (Car_Sales)

        self.MODELS_DIR = self.DEV_DIR / "models"
        if not self.MODELS_DIR.exists():
            # Fallback to project root models dir
            self.MODELS_DIR = self.BASE_DIR / "models"

        self.MLFLOW_DIR = self.DEV_DIR / "mlflow"
        self.MLFLOW_DIR.mkdir(exist_ok=True)

        # Load environment variables from development/.env
        env_path = self.DEV_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"🔑 Loaded environment variables from: {env_path}")
        else:
            print(
                f"⚠️ Warning: .env file not found at {env_path}. Using environment/default fallbacks."
            )

        # Set S3 / MinIO environment variables for MLflow artifact logging
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv(
            "MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000"
        )

        # Set Tracking URI from .env or fallback
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
                with mlflow.start_run(run_name=f"{model_name}") as run:
                    if model_name == "CatBoost":
                        from catboost import CatBoostRegressor

                        model = CatBoostRegressor()
                        model.load_model(model_path)
                    else:
                        model = joblib.load(model_path)

                    artifact_subpath = f"sales_{model_name.lower().replace(' ', '_')}"

                    if model_name in ["Random Forest", "Decision Tree", "XGBoost"]:
                        mlflow.sklearn.log_model(model, artifact_path=artifact_subpath)
                    else:
                        mlflow.log_artifact(str(model_path), artifact_path=artifact_subpath)

                    mlflow.log_metrics(
                        {
                            "R2": float(row.get("R2", 0)),
                            "RMSE": float(row.get("RMSE", 0)),
                            "MAE": float(row.get("MAE", 0)),
                        }
                    )

                    params_file = (
                        sales_dir
                        / "parameters"
                        / f"{model_name.lower().replace(' ', '_')}_best_params.json"
                    )
                    if params_file.exists():
                        with open(params_file, "r") as f:
                            mlflow.log_params(json.load(f))

                    try:
                        mlflow.register_model(
                            f"runs:/{run.info.run_id}/{artifact_subpath}",
                            f"SalesPredictor_{model_name.replace(' ', '_')}",
                        )
                    except Exception as reg_err:
                        print(f"\n⚠️ Registration skipped for {model_name}: {reg_err}")

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
                with mlflow.start_run(run_name=f"quantity_{model_name}") as run:
                    if model_name == "CatBoost":
                        from catboost import CatBoostRegressor

                        model = CatBoostRegressor()
                        model.load_model(model_path)
                    else:
                        model = joblib.load(model_path)

                    artifact_subpath = f"quantity_{model_name.lower().replace(' ', '_')}"

                    if model_name in ["Random Forest", "Decision Tree", "XGBoost"]:
                        mlflow.sklearn.log_model(model, artifact_path=artifact_subpath)
                    else:
                        mlflow.log_artifact(str(model_path), artifact_path=artifact_subpath)

                    mlflow.log_metrics(
                        {
                            "R2": float(row.get("R2", 0)),
                            "RMSE": float(row.get("RMSE", 0)),
                            "MAE": float(row.get("MAE", 0)),
                        }
                    )

                    params_file = (
                        qty_dir
                        / "parameters"
                        / f"{model_name.lower().replace(' ', '_')}_best_params.json"
                    )
                    if params_file.exists():
                        with open(params_file, "r") as f:
                            mlflow.log_params(json.load(f))

                    try:
                        mlflow.register_model(
                            f"runs:/{run.info.run_id}/{artifact_subpath}",
                            f"QuantityPredictor_{model_name.replace(' ', '_')}",
                        )
                    except Exception as reg_err:
                        print(f"\n⚠️ Registration skipped for {model_name}: {reg_err}")

            except Exception as e:
                print(f"\n❌ Failed to load {model_name}: {e}")

    def load_cv_models(self):
        """Load computer vision models"""
        cv_dir = self.MODELS_DIR / "computer_vision_2"
        print(f"\nLoaded CV models from: {cv_dir}")

        if not cv_dir.exists():
            cv_dir = self.MODELS_DIR / "computer_vision"
            print(f"Fallback CV models directory: {cv_dir}")

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
                        mlflow.log_params(
                            {
                                "num_images": metadata.get("num_images", 0),
                                "feature_dimension": metadata.get("feature_dimension", 0),
                            }
                        )

                for file in ["feature_data.csv", "feature_matrix.npy", "brand_mapping.json"]:
                    file_path = cv_dir / file
                    if file_path.exists():
                        mlflow.log_artifact(str(file_path))

                print("✅ Computer Vision artifacts logged successfully!")

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
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
        print(f"📊 MLflow UI: {tracking_uri}")


if __name__ == "__main__":
    loader = MLflowModelLoader()
    loader.load_all()
