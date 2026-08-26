"""
Production Computer Vision Search Service
Loads FAISS index from MinIO
"""

import os
import json
import faiss
import numpy as np
import boto3
import tempfile
import pandas as pd
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models as tv_models
from botocore.client import Config
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Define base directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_DIR = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_DIR)

class ProductionCVSearch:
    def __init__(self):
        self.minio_endpoint = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
        self.access_key = os.getenv('AWS_ACCESS_KEY_ID')
        self.secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.bucket_name = os.getenv('MLFLOW_ARTIFACT_BUCKET', 'mlflow-artifacts')
        
        self._loaded = False
        self._load_from_minio()
    
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
    
    def _load_from_minio(self):
        """Load FAISS index and metadata from MinIO"""
        print("🚀 Loading CV model from MinIO...")
        
        try:
            # Load FAISS index
            index_data = self._read_from_minio('computer_vision/car_index.faiss')
            if index_data:
                with tempfile.NamedTemporaryFile(suffix='.faiss', delete=False) as tmp:
                    tmp.write(index_data)
                    tmp_path = tmp.name
                self.index = faiss.read_index(tmp_path)
                os.unlink(tmp_path)
                print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
            
            # Load metadata
            metadata_data = self._read_from_minio('computer_vision/metadata.json')
            if metadata_data:
                self.metadata = json.loads(metadata_data)
                print(f"✅ Loaded metadata: {self.metadata.get('num_images')} images")
            
            # Load feature data
            feature_data = self._read_from_minio('computer_vision/feature_data.csv')
            if feature_data:
                self.feature_data = pd.read_csv(io.BytesIO(feature_data)).to_dict('records')
                print(f"✅ Loaded feature data: {len(self.feature_data)} records")
            
            self._loaded = True
            
        except Exception as e:
            print(f"❌ Failed to load CV model: {e}")
            self._loaded = False
    
    def _init_feature_extractor(self):
        """Initialize Vision Transformer feature extractor"""
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Identity()
        model = model.to(self.device)
        model.eval()
        
        self.model = model
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        print(f"✅ Feature extractor initialized on {self.device}")
    
    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from image"""
        if not hasattr(self, 'model'):
            self._init_feature_extractor()
        
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        features = features.squeeze().cpu().numpy()
        return features / (np.linalg.norm(features) + 1e-8)
    
    def search(self, image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar cars"""
        if not self._loaded:
            self._load_from_minio()
        
        if not hasattr(self, 'model'):
            self._init_feature_extractor()
        
        # Extract and normalize features
        query_features = self.extract_features(image_path)
        query_features = query_features.reshape(1, -1).astype(np.float32)
        
        # Search FAISS
        distances, indices = self.index.search(query_features, k * 2)
        
        # Get results
        results = []
        seen = set()
        
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.feature_data):
                brand = self.feature_data[idx]['brand']
                if brand in seen:
                    continue
                
                results.append({
                    'brand': brand,
                    'path': self.feature_data[idx]['path'],
                    'similarity': float(score),
                    'rank': len(results) + 1
                })
                seen.add(brand)
                
                if len(results) >= k:
                    break
        
        return results
    
    def get_stats(self) -> Dict:
        """Get search service statistics"""
        return {
            'total_images': self.metadata.get('num_images', 0) if hasattr(self, 'metadata') else 0,
            'feature_dimension': self.metadata.get('feature_dimension', 0) if hasattr(self, 'metadata') else 0,
            'brands': len(self.metadata.get('brands', [])) if hasattr(self, 'metadata') else 0,
            'loaded': self._loaded
        }

# Import io for pandas
import io

# Singleton instance
_cv_search = None

def get_cv_search() -> ProductionCVSearch:
    """Get or create CV search instance"""
    global _cv_search
    if _cv_search is None:
        _cv_search = ProductionCVSearch()
    return _cv_search