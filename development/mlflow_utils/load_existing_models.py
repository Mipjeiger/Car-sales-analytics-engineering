"""
Load existing models from development/models into MLflow
Supports both local and production (Docker/Airflow) environments
"""

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import logging
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class MLflowModelLoader:
    def __init__(self):
        self._detect_environment()
        self._load_env()
        self._setup_mlflow()
        self._validate_paths()

    def _detect_environment(self):
        """Detect if running in Docker/Airflow or local"""
        self.SCRIPT_DIR = Path(__file__).resolve().parent
        
        # Check for Docker environment
        if os.path.exists('/app'):
            self.ENV = 'docker'
            self.DEV_DIR = Path('/app/development')
            self.BASE_DIR = Path('/app')
            logger.info("🐳 Running in Docker environment")
        else:
            self.ENV = 'local'
            self.DEV_DIR = self.SCRIPT_DIR.parent  # development
            self.BASE_DIR = self.DEV_DIR.parent   # project root
            logger.info("💻 Running in local environment")

    def _load_env(self):
        """Load environment variables"""
        env_paths = [
            self.DEV_DIR / '.env',           # development/.env
            self.BASE_DIR / '.env',          # project root/.env
            Path('/app/.env'),               # Docker
        ]
        
        loaded = False
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                loaded = True
                logger.info(f"🔑 Loaded .env from: {env_path}")
                break
        
        if not loaded:
            logger.warning("⚠️ No .env found, using environment variables or defaults")
        
        # Set MinIO/S3 environment variables for MLflow
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        
        # Get tracking URI
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        if self.ENV == 'local':
            self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")

    def _setup_mlflow(self):
        """Setup MLflow connection"""
        mlflow.set_tracking_uri(self.tracking_uri)
        logger.info(f"📡 MLflow Tracking URI: {self.tracking_uri}")

    def _validate_paths(self):
        """Validate and find models directory"""
        possible_paths = [
            self.DEV_DIR / 'models',
            self.BASE_DIR / 'models',
            Path('/app/development/models'),
        ]
        
        for path in possible_paths:
            if path.exists():
                self.MODELS_DIR = path
                logger.info(f"📁 Models directory: {self.MODELS_DIR}")
                break
        else:
            logger.error("❌ Models directory not found!")
            self.MODELS_DIR = self.DEV_DIR / 'models'

        # MLflow directory
        self.MLFLOW_DIR = self.DEV_DIR / 'mlflow_utils'
        self.MLFLOW_DIR.mkdir(exist_ok=True)

    def load_sales_models(self):
        """Load all sales prediction models"""
        sales_dir = self.MODELS_DIR / "sales_prediction"
        metrics_path = sales_dir / "metrics" / "model_metrics.csv"

        if not metrics_path.exists():
            logger.warning(f"⚠️ Metrics file not found: {metrics_path}")
            return

        metrics_df = pd.read_csv(metrics_path)
        model_files = {
            "XGBoost": "xgboost.pkl",
            "Random Forest": "random_forest.pkl",
            "Decision Tree": "decision_tree.pkl",
            "CatBoost": "catboost.cbm",
        }

        logger.info("\n📈 [1/3] Logging Sales Models...")
        for _, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc="Sales Models"):
            model_name = row["Model"]
            model_file = model_files.get(model_name)
            if not model_file:
                continue

            model_path = sales_dir / "models" / model_file
            if not model_path.exists():
                logger.warning(f"⚠️ {model_name} not found: {model_path}")
                continue

            try:
                with mlflow.start_run(run_name=f"sales_{model_name}") as run:
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

                    mlflow.log_metrics({
                        "R2": float(row.get("R2", 0)),
                        "RMSE": float(row.get("RMSE", 0)),
                        "MAE": float(row.get("MAE", 0)),
                    })

                    params_file = sales_dir / "parameters" / f"{model_name.lower().replace(' ', '_')}_best_params.json"
                    if params_file.exists():
                        with open(params_file, "r") as f:
                            mlflow.log_params(json.load(f))

                    try:
                        mlflow.register_model(
                            f"runs:/{run.info.run_id}/{artifact_subpath}",
                            f"SalesPredictor_{model_name.replace(' ', '_')}",
                        )
                    except Exception as reg_err:
                        logger.warning(f"\n⚠️ Registration skipped for {model_name}: {reg_err}")

            except Exception as e:
                logger.error(f"\n❌ Failed to load {model_name}: {e}")

    def load_quantity_models(self):
        """Load all quantity prediction models"""
        qty_dir = self.MODELS_DIR / "quantity_prediction"
        metrics_path = qty_dir / "metrics" / "model_metrics.csv"

        if not metrics_path.exists():
            logger.warning(f"⚠️ Metrics file not found: {metrics_path}")
            return

        metrics_df = pd.read_csv(metrics_path)
        model_files = {
            "XGBoost": "xgboost.pkl",
            "Random Forest": "random_forest.pkl",
            "Decision Tree": "decision_tree.pkl",
            "CatBoost": "catboost.cbm",
        }

        logger.info("\n📦 [2/3] Logging Quantity Models...")
        for _, row in tqdm(metrics_df.iterrows(), total=len(metrics_df), desc="Quantity Models"):
            model_name = row["Model"]
            model_file = model_files.get(model_name)
            if not model_file:
                continue

            model_path = qty_dir / "models" / model_file
            if not model_path.exists():
                logger.warning(f"⚠️ {model_name} not found: {model_path}")
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

                    mlflow.log_metrics({
                        "R2": float(row.get("R2", 0)),
                        "RMSE": float(row.get("RMSE", 0)),
                        "MAE": float(row.get("MAE", 0)),
                    })

                    params_file = qty_dir / "parameters" / f"{model_name.lower().replace(' ', '_')}_best_params.json"
                    if params_file.exists():
                        with open(params_file, "r") as f:
                            mlflow.log_params(json.load(f))

                    try:
                        mlflow.register_model(
                            f"runs:/{run.info.run_id}/{artifact_subpath}",
                            f"QuantityPredictor_{model_name.replace(' ', '_')}",
                        )
                    except Exception as reg_err:
                        logger.warning(f"\n⚠️ Registration skipped for {model_name}: {reg_err}")

            except Exception as e:
                logger.error(f"\n❌ Failed to load {model_name}: {e}")

    def load_cv_models(self):
        """Load computer vision models"""
        cv_dir = self.MODELS_DIR / "computer_vision_2"
        if not cv_dir.exists():
            cv_dir = self.MODELS_DIR / "computer_vision"

        if not cv_dir.exists():
            logger.warning("⚠️ Computer vision directory not found")
            return

        logger.info("\n👁️ [3/3] Logging Computer Vision Artifacts...")
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

                logger.info("✅ Computer Vision artifacts logged successfully!")

        except Exception as e:
            logger.error(f"❌ Failed to load CV models: {e}")

    def load_all(self):
        """Load all models to MLflow"""
        logger.info("🚀 Starting MLflow Loader...")
        logger.info(f"   Environment: {self.ENV}")
        logger.info("=" * 50)

        self.load_sales_models()
        self.load_quantity_models()
        self.load_cv_models()

        logger.info("\n" + "=" * 50)
        logger.info("✅ All models successfully loaded to MLflow!")
        logger.info(f"📊 MLflow UI: {self.tracking_uri}")


if __name__ == "__main__":
    loader = MLflowModelLoader()
    loader.load_all()