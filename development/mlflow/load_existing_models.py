"""
Load existing models from MinIO into MLflow
Supports both local and production (Docker/Airflow) environments
"""

import os
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import io
import boto3
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv
from botocore.client import Config
import tempfile


class MLflowModelLoader:
    def __init__(self):
<<<<<<< Updated upstream
        # Resolve directory hierarchy dynamically relative to this script
        # Script location: development/mlflow/load_existing_models.py
        self.SCRIPT_DIR = Path(__file__).resolve().parent  # development/mlflow
        self.DEV_DIR = self.SCRIPT_DIR.parent               # development
        self.BASE_DIR = self.DEV_DIR.parent                # project root (Car_Sales)
        
        self.MODELS_DIR = self.DEV_DIR / "models"
        if not self.MODELS_DIR.exists():
            self.MODELS_DIR = self.BASE_DIR / "models"  # Fallback to project root models dir

        self.MLFLOW_DIR = self.DEV_DIR / "mlflow"
        self.MLFLOW_DIR.mkdir(exist_ok=True)

        # Load environment variables from development/.env
        env_path = self.DEV_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"🔑 Loaded environment variables from: {env_path}")
        else:
            print(f"⚠️ Warning: .env file not found at {env_path}. Using environment/default fallbacks.")

        # Set S3 / MinIO environment variables for MLflow artifact logging
        os.environ["AWS_ACCESS_KEY_ID"] = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
        os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
=======
        self._detect_environment()        
        self._load_env()
        self._setup_minio()
        self._setup_mlflow()
        self._validate_bucket()

    def _detect_environment(self):
        """Detect if running in Docker/Airflow or local"""
        self.SCRIPT_DIR = Path(__file__).resolve().parent
        
        if os.path.exists('/app'):
            self.ENV = 'docker'
            self.DEV_DIR = Path('/app/development')
            print("🐳 Running in Docker environment")
        else:
            self.ENV = 'local'
            self.DEV_DIR = self.SCRIPT_DIR.parent
            print("💻 Running in local environment")

    def _load_env(self):
        """Load environment variables"""
        env_paths = [
            self.DEV_DIR / '.env',
            Path('/app/.env'),
        ]
        
        loaded = False
        for env_path in env_paths:
            if env_path.exists():
                load_dotenv(dotenv_path=env_path)
                loaded = True
                print(f"🔑 Loaded .env from: {env_path}")
                break
        
        if not loaded:
            print("⚠️ No .env found, using defaults")
        
        # MinIO/S3 configuration
        self.minio_endpoint = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        self.access_key = os.getenv("AWS_ACCESS_KEY_ID")
        self.secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.bucket_name = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")
        
        # MLflow tracking URI
        self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        if self.ENV == 'local':
            self.tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")

    def _setup_minio(self):
        """Setup MinIO/S3 client"""
        self.s3 = boto3.client(
            's3',
            endpoint_url=self.minio_endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
        print(f"🪣 MinIO endpoint: {self.minio_endpoint}")
>>>>>>> Stashed changes

    def _setup_mlflow(self):
        """Setup MLflow connection"""
        # Set environment variables for MLflow
        os.environ["AWS_ACCESS_KEY_ID"] = self.access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = self.secret_key
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.minio_endpoint
        
        mlflow.set_tracking_uri(self.tracking_uri)
        print(f"📡 MLflow Tracking URI: {self.tracking_uri}")

    def _validate_bucket(self):
        """Validate MinIO bucket exists"""
        try:
            self.s3.head_bucket(Bucket=self.bucket_name)
            print(f"✅ Bucket '{self.bucket_name}' exists")
        except:
            print(f"⚠️ Bucket '{self.bucket_name}' not found, creating...")
            self.s3.create_bucket(Bucket=self.bucket_name)

    def _read_from_minio(self, key):
        """Read file from MinIO"""
        try:
            response = self.s3.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except Exception as e:
            print(f"⚠️ Error reading {key}: {e}")
            return None

    def _list_models_from_minio(self, prefix):
        """List models in MinIO"""
        models = []
        try:
            response = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=prefix)
            for obj in response.get('Contents', []):
                key = obj['Key']
                if key.endswith(('.pkl', '.cbm')):
                    models.append({
                        'key': key,
                        'model_name': key.split('/')[-1].replace('.pkl', '').replace('.cbm', ''),
                        'size': obj['Size'],
                        'last_modified': obj['LastModified']
                    })
        except Exception as e:
            print(f"⚠️ Error listing models: {e}")
        return models

    def load_sales_models(self):
        """Load all sales prediction models from MinIO"""
        print("\n📈 [1/3] Loading Sales Models from MinIO...")
        
        # Get list of models from MinIO
        prefix = "sales_prediction/models/"
        models = self._list_models_from_minio(prefix)
        
        if not models:
            print("⚠️ No sales models found in MinIO")
            return
        
        for model in tqdm(models, desc="Sales Models"):
            model_name = model['model_name']
            
            # Map to display name
            display_names = {
                'xgboost': 'XGBoost',
                'random_forest': 'Random Forest',
                'decision_tree': 'Decision Tree',
                'catboost': 'CatBoost'
            }
            display_name = display_names.get(model_name.lower(), model_name)
            
            try:
                # Read model from MinIO
                model_bytes = self._read_from_minio(model['key'])
                if not model_bytes:
                    continue
                
                with mlflow.start_run(run_name=f"sales_{display_name}") as run:
                    # Load model based on type
                    if model['key'].endswith('.cbm'):
                        from catboost import CatBoostRegressor
<<<<<<< Updated upstream
                        model = CatBoostRegressor()
                        model.load_model(model_path)
=======
                        with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
                            tmp.write(model_bytes)
                            tmp_path = tmp.name
                        model_obj = CatBoostRegressor()
                        model_obj.load_model(tmp_path)
                        os.unlink(tmp_path)
                        artifact_subpath = f"sales_{model_name.lower()}"
>>>>>>> Stashed changes
                    else:
                        model_obj = joblib.load(io.BytesIO(model_bytes))
                        artifact_subpath = f"sales_{model_name.lower()}"
                    
                    # Log model
                    if model['key'].endswith('.cbm'):
                        mlflow.log_artifact(tmp_path, artifact_path=artifact_subpath)
                    else:
<<<<<<< Updated upstream
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

=======
                        mlflow.sklearn.log_model(model_obj, artifact_path=artifact_subpath)
                    
                    # Try to load metrics
                    metrics_key = f"sales_prediction/metrics/training_results.csv"
                    metrics_data = self._read_from_minio(metrics_key)
                    if metrics_data:
                        df = pd.read_csv(io.BytesIO(metrics_data))
                        matching = df[df['Model'] == display_name]
                        if not matching.empty:
                            mlflow.log_metrics({
                                "R2": float(matching.iloc[0].get('R2', 0)),
                                "RMSE": float(matching.iloc[0].get('RMSE', 0)),
                                "MAE": float(matching.iloc[0].get('MAE', 0)),
                            })
                    
                    # Register model
>>>>>>> Stashed changes
                    try:
                        mlflow.register_model(
                            f"runs:/{run.info.run_id}/{artifact_subpath}",
                            f"SalesPredictor_{display_name.replace(' ', '_')}",
                        )
                        print(f"✅ Registered Sales {display_name}")
                    except Exception as reg_err:
                        print(f"⚠️ Registration skipped for {display_name}: {reg_err}")
                        
            except Exception as e:
                print(f"❌ Failed to load {display_name}: {e}")

    def load_quantity_models(self):
        """Load all quantity prediction models from MinIO"""
        print("\n📦 [2/3] Loading Quantity Models from MinIO...")
        
        prefix = "quantity_prediction/models/"
        models = self._list_models_from_minio(prefix)
        
        if not models:
            print("⚠️ No quantity models found in MinIO")
            return
        
        for model in tqdm(models, desc="Quantity Models"):
            model_name = model['model_name']
            
            display_names = {
                'xgboost': 'XGBoost',
                'random_forest': 'Random Forest',
                'decision_tree': 'Decision Tree',
                'catboost': 'CatBoost'
            }
            display_name = display_names.get(model_name.lower(), model_name)
            
            try:
                model_bytes = self._read_from_minio(model['key'])
                if not model_bytes:
                    continue
                
                with mlflow.start_run(run_name=f"quantity_{display_name}") as run:
                    if model['key'].endswith('.cbm'):
                        from catboost import CatBoostRegressor
<<<<<<< Updated upstream
                        model = CatBoostRegressor()
                        model.load_model(model_path)
=======
                        with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
                            tmp.write(model_bytes)
                            tmp_path = tmp.name
                        model_obj = CatBoostRegressor()
                        model_obj.load_model(tmp_path)
                        os.unlink(tmp_path)
                        artifact_subpath = f"quantity_{model_name.lower()}"
>>>>>>> Stashed changes
                    else:
                        model_obj = joblib.load(io.BytesIO(model_bytes))
                        artifact_subpath = f"quantity_{model_name.lower()}"
                    
                    if model['key'].endswith('.cbm'):
                        mlflow.log_artifact(tmp_path, artifact_path=artifact_subpath)
                    else:
<<<<<<< Updated upstream
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

=======
                        mlflow.sklearn.log_model(model_obj, artifact_path=artifact_subpath)
                    
                    # Load metrics
                    metrics_key = f"quantity_prediction/metrics/training_results.csv"
                    metrics_data = self._read_from_minio(metrics_key)
                    if metrics_data:
                        df = pd.read_csv(io.BytesIO(metrics_data))
                        matching = df[df['Model'] == display_name]
                        if not matching.empty:
                            mlflow.log_metrics({
                                "R2": float(matching.iloc[0].get('R2', 0)),
                                "RMSE": float(matching.iloc[0].get('RMSE', 0)),
                                "MAE": float(matching.iloc[0].get('MAE', 0)),
                            })
                    
>>>>>>> Stashed changes
                    try:
                        mlflow.register_model(
                            f"runs:/{run.info.run_id}/{artifact_subpath}",
                            f"QuantityPredictor_{display_name.replace(' ', '_')}",
                        )
                        print(f"✅ Registered Quantity {display_name}")
                    except Exception as reg_err:
                        print(f"⚠️ Registration skipped for {display_name}: {reg_err}")
                        
            except Exception as e:
                print(f"❌ Failed to load {display_name}: {e}")

    def load_cv_models(self):
        """Load computer vision models from MinIO"""
        print("\n👁️ [3/3] Loading Computer Vision Models from MinIO...")
        
        prefix = "computer_vision/"
        
        try:
<<<<<<< Updated upstream
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

=======
            with mlflow.start_run(run_name="computer_vision_faiss") as run:
                # Download FAISS index
                index_data = self._read_from_minio(f"{prefix}car_index.faiss")
                if index_data:
                    with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp:
                        tmp.write(index_data)
                        tmp_path = tmp.name
                    mlflow.log_artifact(tmp_path)
                    os.unlink(tmp_path)
                    print("✅ Logged FAISS index")
                
                # Log metadata
                metadata_data = self._read_from_minio(f"{prefix}metadata.json")
                if metadata_data:
                    metadata = json.loads(metadata_data)
                    mlflow.log_params({
                        "num_images": metadata.get("num_images", 0),
                        "feature_dimension": metadata.get("feature_dimension", 0),
                        "brands": len(metadata.get("brands", []))
                    })
                    print("✅ Logged metadata")
                
                # Log feature files
>>>>>>> Stashed changes
                for file in ["feature_data.csv", "feature_matrix.npy", "brand_mapping.json"]:
                    file_data = self._read_from_minio(f"{prefix}{file}")
                    if file_data:
                        with tempfile.NamedTemporaryFile(suffix=f'.{file.split(".")[-1]}', delete=False) as tmp:
                            tmp.write(file_data)
                            tmp_path = tmp.name
                        mlflow.log_artifact(tmp_path)
                        os.unlink(tmp_path)
                        print(f"✅ Logged {file}")
                
                # Register CV model
                try:
                    mlflow.register_model(
                        f"runs:/{run.info.run_id}",
                        "ComputerVision_FAISS",
                    )
                    print("✅ Registered ComputerVision_FAISS")
                except Exception as reg_err:
                    print(f"⚠️ Registration skipped: {reg_err}")
                    
        except Exception as e:
            print(f"❌ Failed to load CV models: {e}")

    def load_all(self):
        """Load all models from MinIO to MLflow"""
        print("🚀 Starting MLflow Loader from MinIO...")
        print(f"   Environment: {self.ENV}")
        print(f"   Bucket: {self.bucket_name}")
        print("=" * 50)

        self.load_sales_models()
        self.load_quantity_models()
        self.load_cv_models()

        print("\n" + "=" * 50)
        print("✅ All models successfully loaded to MLflow from MinIO!")
        print(f"📊 MLflow UI: {self.tracking_uri}")
        print(f"🪣 MinIO Console: {self.minio_endpoint.replace(':9000', ':9001')}")


if __name__ == "__main__":
    loader = MLflowModelLoader()
    loader.load_all()