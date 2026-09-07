"""
Production Prediction Service
Loads models from MinIO and serves predictions
"""

import os
import io
import joblib
import json
import boto3
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from botocore.client import Config
from typing import Dict, Any, Optional
from dotenv import load_dotenv
import tempfile

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_DIR = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_DIR)

class ProductionPredictor:
    def __init__(self):
        # MinIO Configuration
        self.minio_endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.bucket_name = os.getenv('MLFLOW_ARTIFACT_BUCKET', 'mlflow-artifacts')
        
        # MLflow Configuration
        self.tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
        
        self.sales_models = {}
        self.qty_models = {}
        self.scalers = {}
        self._loaded = False
        
        # Lazy loading - models loaded on first prediction
        self._load_models()
    
    def _get_minio_client(self):
        """Get MinIO client"""
        return boto3.client(
            's3',
            endpoint_url=self.minio_endpoint,
            aws_access_key_id=self.access_key,
            aws_secret_access_key=self.secret_key,
            config=Config(signature_version='s3v4'),
            region_name='us-east-1'
        )
    
    def _read_from_minio(self, key: str) -> Optional[bytes]:
        """Read file from MinIO"""
        try:
            s3 = self._get_minio_client()
            response = s3.get_object(Bucket=self.bucket_name, Key=key)
            return response['Body'].read()
        except Exception as e:
            print(f"⚠️ Error reading {key}: {e}")
            return None
    
    def _load_model_from_minio(self, model_type: str, model_name: str):
        """Load a single model from MinIO"""
        key = f"{model_type}_prediction/models/{model_name.lower().replace(' ', '_')}.pkl"
        
        # Try CatBoost
        if model_name == 'CatBoost':
            key = f"{model_type}_prediction/models/catboost.cbm"
            model_bytes = self._read_from_minio(key)
            if model_bytes:
                from catboost import CatBoostRegressor
                with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
                    tmp.write(model_bytes)
                    tmp_path = tmp.name
                model = CatBoostRegressor()
                model.load_model(tmp_path)
                os.unlink(tmp_path)
                return model
        
        # Try sklearn models
        model_bytes = self._read_from_minio(key)
        if model_bytes:
            return joblib.load(io.BytesIO(model_bytes))
        
        return None
    
    def _load_scaler_from_minio(self, model_type: str):
        """Load scaler from MinIO"""
        key = f"{model_type}_prediction/scalers/feature_scaler.pkl"
        model_bytes = self._read_from_minio(key)
        if model_bytes:
            return joblib.load(io.BytesIO(model_bytes))
        return None
    
    def _load_models(self):
        """Load models from MinIO"""
        print("🚀 Loading production models from MinIO...")
        
        model_names = ['XGBoost', 'Random Forest', 'Decision Tree', 'CatBoost']
        
        # Load sales models
        for name in model_names:
            model = self._load_model_from_minio('sales', name)
            if model:
                self.sales_models[name] = model
                print(f"✅ Loaded Sales {name} from MinIO")
        
        # Load quantity models
        for name in model_names:
            model = self._load_model_from_minio('quantity', name)
            if model:
                self.qty_models[name] = model
                print(f"✅ Loaded Quantity {name} from MinIO")
        
        # Load scalers
        for model_type in ['sales', 'quantity']:
            scaler = self._load_scaler_from_minio(model_type)
            if scaler:
                self.scalers[model_type] = scaler
                print(f"✅ Loaded {model_type} scaler from MinIO")
        
        # Try MLflow as fallback if MinIO fails
        if not self.sales_models and not self.qty_models:
            print("⚠️ No models found in MinIO, trying MLflow...")
            self._load_from_mlflow()
        
        self._loaded = True
    
    def _load_from_mlflow(self):
        """Load models from MLflow as fallback"""
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            client = mlflow.tracking.MlflowClient()
            
            # Get registered models
            models = client.search_registered_models()
            for model in models:
                if 'SalesPredictor' in model.name:
                    name = model.name.replace('SalesPredictor_', '')
                    latest = client.get_latest_versions(model.name, stages=["Production"])
                    if latest:
                        model_obj = mlflow.sklearn.load_model(f"models:/{model.name}/Production")
                        self.sales_models[name] = model_obj
                        print(f"✅ Loaded Sales {name} from MLflow")
                
                elif 'QuantityPredictor' in model.name:
                    name = model.name.replace('QuantityPredictor_', '')
                    latest = client.get_latest_versions(model.name, stages=["Production"])
                    if latest:
                        model_obj = mlflow.sklearn.load_model(f"models:/{model.name}/Production")
                        self.qty_models[name] = model_obj
                        print(f"✅ Loaded Quantity {name} from MLflow")
        except Exception as e:
            print(f"⚠️ MLflow fallback failed: {e}")
    
    def _prepare_features(self, features: Dict, model_type: str) -> pd.DataFrame:
        """Prepare features for prediction"""
        if isinstance(features, dict):
            df = pd.DataFrame([features])
        else:
            df = features
        
        # Apply scaler if available
        if model_type in self.scalers:
            scaler = self.scalers[model_type]
            # Get feature columns (assuming they match)
            if hasattr(scaler, 'feature_names_in_'):
                cols = scaler.feature_names_in_
                df = df[cols] if all(c in df.columns for c in cols) else df
            try:
                scaled = scaler.transform(df)
                return pd.DataFrame(scaled, columns=df.columns)
            except:
                pass
        
        return df
    
    def predict_sales(self, features: Dict, model_name: str = 'XGBoost') -> float:
        """Predict sales using selected model"""
        if not self._loaded:
            self._load_models()
        
        model = self.sales_models.get(model_name)
        if not model:
            raise ValueError(f"Sales model '{model_name}' not found. Available: {list(self.sales_models.keys())}")
        
        # Prepare features
        prepared = self._prepare_features(features, 'sales')
        
        # Predict
        try:
            prediction = model.predict(prepared)
            return float(prediction[0])
        except Exception as e:
            print(f"⚠️ Prediction failed: {e}")
            # Try without scaler
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            else:
                df = features
            prediction = model.predict(df)
            return float(prediction[0])
    
    def predict_quantity(self, features: Dict, model_name: str = 'XGBoost') -> float:
        """Predict quantity using selected model"""
        if not self._loaded:
            self._load_models()
        
        model = self.qty_models.get(model_name)
        if not model:
            raise ValueError(f"Quantity model '{model_name}' not found. Available: {list(self.qty_models.keys())}")
        
        prepared = self._prepare_features(features, 'quantity')
        
        try:
            prediction = model.predict(prepared)
            return float(prediction[0])
        except:
            if isinstance(features, dict):
                df = pd.DataFrame([features])
            else:
                df = features
            prediction = model.predict(df)
            return float(prediction[0])
    
    def list_models(self) -> Dict:
        """List available models"""
        return {
            'sales': list(self.sales_models.keys()),
            'quantity': list(self.qty_models.keys()),
            'source': 'MinIO' if any(self.sales_models) else 'MLflow'
        }
    
    def get_model_info(self, model_type: str, model_name: str) -> Dict:
        """Get model information"""
        models = self.sales_models if model_type == 'sales' else self.qty_models
        if model_name in models:
            return {
                'model_name': model_name,
                'model_type': model_type,
                'loaded': True,
                'source': 'MinIO'
            }
        return {'loaded': False}

# Singleton instance
_predictor = None

def get_predictor() -> ProductionPredictor:
    """Get or create predictor instance"""
    global _predictor
    if _predictor is None:
        _predictor = ProductionPredictor()
    return _predictor