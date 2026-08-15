import faiss
import numpy as np
import json
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn as nn
import logging
from torchvision import transforms, models as tv_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CVSearchService:
    """Service to perform computer vision search using FAISS and pre-trained models"""
    def __init__(self):
        self.BASE_DIR = self._find_project_root()
        self.MODELS_DIR = self.BASE_DIR / "models" / "computer_vision_2"
        
        index_path = self.MODELS_DIR / "car_index.faiss"
        metadata_path = self.MODELS_DIR / "metadata.json"

        # Sanity check before loading
        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at expected path: {index_path}")

        # Load FAISS index
        self.index = faiss.read_index(str(index_path))

        # Load metadata
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Load feature data
        self.feature_data = self._load_feature_data()

        # Init feature extractor model
        self.extractor = self._init_extractor()

        logger.info(f"✅ CV Search Service initialized with {self.index.ntotal} images")

    @staticmethod
    def _find_project_root() -> Path:
        """Find project root by looking for 'models' directory up the path tree"""
        current = Path(__file__).resolve().parent
        for parent in [current] + list(current.parents):
            if (parent / "models").exists():
                return parent
        
        return Path(__file__).resolve().parents[3] # Fallback: 4 levels up if not found explicitly

    def _load_feature_data(self) -> List[Dict]:
        """Load feature data from csv"""
        import pandas as pd
        df = pd.read_csv(self.MODELS_DIR / "feature_data.csv")
        return df.to_dict(orient="records")

    def _init_extractor(self):
        """Initialize Vision Transformer"""
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = tv_models.vit_b_16(weights=tv_models.ViT_B_16_Weights.DEFAULT)
        model.heads = nn.Identity()  # Remove classification head
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

        return {
            'device': device,
            'model': model,
            'transform': transform
        }

    def extract_features(self, image_path: str) -> np.ndarray:
        """Extract features from image"""
        image = Image.open(image_path).convert("RGB")

        device = self.extractor['device']
        transform = self.extractor['transform']
        model = self.extractor['model']

        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            features = model(image_tensor).cpu().numpy().flatten()

        # Normalize features
        features = (features / np.linalg.norm(features) + 1e-8)

        return features

    def search(self, image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search similar cars"""
        
        # Extract features
        query_features = self.extract_features(image_path)
        query_features = query_features.reshape(1, -1).astype(np.float32)
        
        # Search FAISS
        distances, indices = self.index.search(query_features, k * 2)  # Get extra for filtering
        
        # Get results
        results = []
        seen_brands = set()
        
        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.feature_data):
                brand = self.feature_data[idx]['brand']
                
                # Avoid duplicates
                if brand in seen_brands:
                    continue
                
                result = {
                    'brand': brand,
                    'path': self.feature_data[idx]['path'],
                    'similarity': float(score),
                    'rank': len(results) + 1
                }
                
                results.append(result)
                seen_brands.add(brand)
                
                if len(results) >= k:
                    break
        
        return results

    def search_by_image(self, image_path: str, k: int = 5) -> List[Dict[str, Any]]:
        """Public search method"""
        try:
            return self.search(image_path, k)
        except Exception as e:
            print(f"Search error: {e}")
            return []

# Singleton instance
cv_search = None

def get_cv_search() -> CVSearchService:
    """Get or create CV Search service"""
    global cv_search
    if cv_search is None:
        cv_search = CVSearchService()
    return cv_search