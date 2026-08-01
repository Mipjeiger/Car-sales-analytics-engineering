"""
Load existing models from development/models into MLflow
"""

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import json
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

class MLflowModelLoader:
    def __init__(self):
        self.BASE_DIR = Path.cwd()
        self.MODELS_DIR = self.BASE_DIR / 'development' / 'models'
        self.MLFLOW_DIR = self.BASE_DIR / 'development' / 'mlflow'
        
        # Setup MLflow with MinIO
        self.MLFLOW_DIR.mkdir(exist_ok=True)
        mlflow.set_tracking_uri("http://mlflow:5000")
        
    def load_sales_models(self):
        """Load all sales prediction models"""
        sales_dir = self.MODELS_DIR / 'sales_prediction'
        metrics_path = sales_dir / 'metrics' / 'model_metrics.csv'
        
        if not metrics_path.exists():
            print(f"⚠️ Metrics file not found: {metrics_path}")
            return
        
        metrics_df = pd.read_csv(metrics_path)
        model_files = {
            'XGBoost': 'xgboost.pkl',
            'Random Forest': 'random_forest.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'CatBoost': 'catboost.cbm'
        }
        
        for _, row in metrics_df.iterrows():
            model_name = row['Model']
            model_file = model_files.get(model_name)
            if not model_file:
                continue
                
            model_path = sales_dir / 'models' / model_file
            if not model_path.exists():
                print(f"⚠️ {model_name} not found: {model_path}")
                continue
            
            try:
                with mlflow.start_run(run_name=f"sales_{model_name}"):
                    # Load model
                    if model_name == 'CatBoost':
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor()
                        model.load_model(model_path)
                    else:
                        model = joblib.load(model_path)
                    
                    # Log model
                    mlflow.sklearn.log_model(model, f"sales_{model_name}")
                    
                    # Log metrics
                    mlflow.log_metrics({
                        'R2': float(row.get('R2', 0)),
                        'RMSE': float(row.get('RMSE', 0)),
                        'MAE': float(row.get('MAE', 0))
                    })
                    
                    # Log parameters
                    params_file = sales_dir / 'parameters' / f"{model_name.lower().replace(' ', '_')}_best_params.json"
                    if params_file.exists():
                        with open(params_file, 'r') as f:
                            mlflow.log_params(json.load(f))
                    
                    # Register model
                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/sales_{model_name}",
                        f"SalesPredictor_{model_name}"
                    )
                    
                    print(f"✅ Sales {model_name} loaded")
                    
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
    
    def load_quantity_models(self):
        """Load all quantity prediction models"""
        qty_dir = self.MODELS_DIR / 'quantity_prediction'
        metrics_path = qty_dir / 'metrics' / 'model_metrics.csv'
        
        if not metrics_path.exists():
            print(f"⚠️ Metrics file not found: {metrics_path}")
            return
        
        metrics_df = pd.read_csv(metrics_path)
        model_files = {
            'XGBoost': 'xgboost.pkl',
            'Random Forest': 'random_forest.pkl',
            'Decision Tree': 'decision_tree.pkl',
            'CatBoost': 'catboost.cbm'
        }
        
        for _, row in metrics_df.iterrows():
            model_name = row['Model']
            model_file = model_files.get(model_name)
            if not model_file:
                continue
                
            model_path = qty_dir / 'models' / model_file
            if not model_path.exists():
                print(f"⚠️ {model_name} not found: {model_path}")
                continue
            
            try:
                with mlflow.start_run(run_name=f"quantity_{model_name}"):
                    if model_name == 'CatBoost':
                        from catboost import CatBoostRegressor
                        model = CatBoostRegressor()
                        model.load_model(model_path)
                    else:
                        model = joblib.load(model_path)
                    
                    mlflow.sklearn.log_model(model, f"quantity_{model_name}")
                    
                    mlflow.log_metrics({
                        'R2': float(row.get('R2', 0)),
                        'RMSE': float(row.get('RMSE', 0)),
                        'MAE': float(row.get('MAE', 0))
                    })
                    
                    params_file = qty_dir / 'parameters' / f"{model_name.lower().replace(' ', '_')}_best_params.json"
                    if params_file.exists():
                        with open(params_file, 'r') as f:
                            mlflow.log_params(json.load(f))
                    
                    mlflow.register_model(
                        f"runs:/{mlflow.active_run().info.run_id}/quantity_{model_name}",
                        f"QuantityPredictor_{model_name}"
                    )
                    
                    print(f"✅ Quantity {model_name} loaded")
                    
            except Exception as e:
                print(f"❌ Failed to load {model_name}: {e}")
    
    def load_cv_models(self):
        """Load computer vision models"""
        cv_dir = self.MODELS_DIR / 'computer_vision_2'
        if not cv_dir.exists():
            cv_dir = self.MODELS_DIR / 'computer_vision'
        
        if not cv_dir.exists():
            print("⚠️ Computer vision directory not found")
            return
        
        try:
            with mlflow.start_run(run_name="computer_vision_faiss"):
                # Log FAISS index
                index_path = cv_dir / 'car_index.faiss'
                if index_path.exists():
                    mlflow.log_artifact(str(index_path))
                
                # Log metadata
                metadata_path = cv_dir / 'metadata.json'
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        mlflow.log_params({
                            'num_images': metadata.get('num_images', 0),
                            'feature_dimension': metadata.get('feature_dimension', 0)
                        })
                
                # Log feature data
                for file in ['feature_data.csv', 'feature_matrix.npy', 'brand_mapping.json']:
                    file_path = cv_dir / file
                    if file_path.exists():
                        mlflow.log_artifact(str(file_path))
                
                # Register CV model
                mlflow.register_model(
                    f"runs:/{mlflow.active_run().info.run_id}",
                    "ComputerVision_FAISS"
                )
                
                print(f"✅ Computer Vision model loaded")
                
        except Exception as e:
            print(f"❌ Failed to load CV models: {e}")
    
    def load_all(self):
        """Load all models to MLflow"""
        print("🚀 Loading models to MLflow...")
        print("="*50)
        
        self.load_sales_models()
        print("")
        self.load_quantity_models()
        print("")
        self.load_cv_models()
        
        print("\n✅ All models loaded to MLflow!")
        print(f"📊 MLflow UI: http://localhost:5003")

if __name__ == "__main__":
    loader = MLflowModelLoader()
    loader.load_all()