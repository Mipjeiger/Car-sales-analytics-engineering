"""
Serve models from MLflow
"""

import mlflow
import json
import numpy as np
import pandas as pd
from pathlib import Path

class MLflowModelServing:
    def __init__(self):
        self.BASE_DIR = Path.cwd()
        self.MLFLOW_DIR = self.BASE_DIR / 'development' / 'mlflow'
        
        mlflow.set_tracking_uri("http://mlflow:5000")
        self.load_models()
    
    def load_models(self):
        """Load models from MLflow registry"""
        self.models = {'sales': {}, 'quantity': {}, 'cv': None}
        
        client = mlflow.tracking.MlflowClient()
        registered_models = client.search_registered_models()
        
        for model in registered_models:
            if 'SalesPredictor' in model.name:
                latest = client.get_latest_versions(model.name, stages=["Production"])
                if latest:
                    model_key = model.name.replace('SalesPredictor_', '')
                    self.models['sales'][model_key] = {
                        'model': mlflow.sklearn.load_model(f"models:/{model.name}/Production"),
                        'version': latest[0].version
                    }
            elif 'QuantityPredictor' in model.name:
                latest = client.get_latest_versions(model.name, stages=["Production"])
                if latest:
                    model_key = model.name.replace('QuantityPredictor_', '')
                    self.models['quantity'][model_key] = {
                        'model': mlflow.sklearn.load_model(f"models:/{model.name}/Production"),
                        'version': latest[0].version
                    }
            elif 'ComputerVision' in model.name:
                self.models['cv'] = {'version': '1'}
        
        print(f"✅ Loaded {len(self.models['sales'])} sales models")
        print(f"✅ Loaded {len(self.models['quantity'])} quantity models")
    
    def predict_sales(self, features, model_name='XGBoost'):
        if model_name not in self.models['sales']:
            raise ValueError(f"Model {model_name} not found")
        return self.models['sales'][model_name]['model'].predict(features)
    
    def predict_quantity(self, features, model_name='XGBoost'):
        if model_name not in self.models['quantity']:
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