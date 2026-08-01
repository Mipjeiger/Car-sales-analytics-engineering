"""
Serve models from MLflow
"""

import os
import mlflow
import mlflow.pyfunc
import json
from pathlib import Path
from dotenv import load_dotenv

class MLflowModelServing:
    def __init__(self):
        self.BASE_DIR = Path.cwd().resolve().parent
        self.DEV_DIR = self.BASE_DIR.parent # development

        # Load environment variables
        env_path = self.DEV_DIR / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        # Set tracking URI dynamically
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5003")
        mlflow.set_tracking_uri(tracking_uri)
        print(f"📡 Connected to MLflow Tracking Server at: {tracking_uri}")
    
    def load_models(self):
        """Load models from MLflow registry"""
        self.models = {'sales': {}, 'quantity': {}, 'cv': None}
        
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        
        for model in registered_models:
            # Get all versions for the model regardless of stage
            versions = client.search_model_versions(f"name='{model.name}'")
            if not versions:
                continue

            # Retrieve the latest numerical version
            latest_version_obj = max(versions, key=lambda v: int(v.version))
            latest_version = latest_version_obj.version
            model_uri = f"models:/{model.name}/{latest_version}"

            # Safely attempt to load model object using MLFlow pyfunc
            try:
                loaded_model = mlflow.pyfunc.load_model(model_uri)
            except Exception:
                loaded_model = None

            # Ensure all models witin self.models
            if 'SalesPredictor' in model.name:
                model_key = model.name.replace('SalesPredictor_', '')
                self.models['sales'][model_key] = {
                    'model': loaded_model,
                    'version': latest_version
                }
            elif 'QuantityPredictor' in model.name:
                model_key = model.name.replace('QuantityPredictor_', '')
                self.models['quantity'][model_key] = {
                    'model': loaded_model,
                    'version': latest_version
                }
            elif 'ComputerVisionModel' in model.name:
                self.models['cv'] = {'version': latest_version}

        print(f"✅ Loaded {len(self.models['sales'])} sales models")
        print(f"✅ Loaded {len(self.models['quantity'])} quantity models")
        print(f"✅ Loaded {len(self.models['cv'])} computer vision models")
    
    def predict_sales(self, features, model_name='XGBoost'):
        """Predict sales using the specified model"""
        if model_name not in self.models['sales'] or not self.models['sales'][model_name]['model']:
            raise ValueError(f"Model {model_name} not found")
        return self.models['sales'][model_name]['model'].predict(features)
    
    def predict_quantity(self, features, model_name='XGBoost'):
        """Predict quantity using the specified model"""
        if model_name not in self.models['quantity'] or not self.models['quantity'][model_name]['model']:
            raise ValueError(f"Model {model_name} not found")
        return self.models['quantity'][model_name]['model'].predict(features)
    
    def get_model_info(self):
        return {
            'sales_models': {k: v['version'] for k, v in self.models['sales'].items()},
            'quantity_models': {k: v['version'] for k, v in self.models['quantity'].items()}
        }

if __name__ == "__main__":
    serving = MLflowModelServing()
    print("\n📊 Model Registry Info:")
    print(json.dumps(serving.get_model_info(), indent=2))