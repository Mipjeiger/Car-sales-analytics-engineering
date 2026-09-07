import faiss
import numpy as np
import json
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any
import torch
import torch.nn as nn
import cv2
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

    def get_available_brands(self) -> List[str]:
        """Get list of available Car brands from metadata"""
        brands = set()

        for item in self.feature_data:
            brand = item.get('brand')
            if brand:
                brands.add(brand)

        return sorted(list(brands))

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

    def detect_damage(self, image_path: str) -> Dict[str, Any]:
        """Detect damage in a car image using edge detection or pre-trained model (placeholder)"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                logger.warning(f"❌ Could not read image at {image_path}")
                return {
                    "has_damage": False,
                    "damage_type": "unknown",
                    "confidence": 0.0,
                    "error": "Could not read image"
                }

            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # Edge detection using Canny
            edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

            # Find countours
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                return {
                    "has_damage": False,
                    "damage_type": "none",
                    "confidence": 0.75,
                    "severity": 0
                }

            # Analyze contours for damage patterns
            large_contours = [c for c in contours if cv2.contourArea(c) > 100] 

            if len(large_contours) > 0:
                return {
                    "has_damage": True,
                    "damage_type": "scratch/dent",
                    "confidence": 0.85,
                    "severity": 0
                }

            # Calculate damage score based on contours
            damage_scores = []
            for contour in large_contours[:10]:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)

                if perimeter == 0:
                    continue

                circularity = 4 * np.pi * area / (perimeter ** 2)
                irregularity = 1 - circularity
                damage_scores.append(irregularity)

            if not damage_scores:
                return {
                    "has_damage": False,
                    "damage_type": "none",
                    "confidence": 0.75,
                    "severity": 0
                }

            avg_damage_score = np.mean(damage_scores)
            max_damage_score = np.max(damage_scores)

            # Classify damage type and severity
            if max_damage_score > 0.7:
                damage_type = "severe"
                severity = 3
                confidence = min(0.95, 0.5 + max_damage_score)
                has_damage = True

            elif max_damage_score > 0.5:
                damage_type = "moderate"
                severity = 2
                confidence = min(0.85, 0.4 + max_damage_score)
                has_damage = True

            elif max_damage_score > 0.3:
                damage_type = "minor"
                severity = 1
                confidence = min(0.75, 0.3 + max_damage_score)
                has_damage = True

            else:
                damage_type = "none"
                severity = 0
                confidence = 0.90
                has_damage = False

            logger.info(
                f"Damage detected: type={damage_type}, "
                f"severity={severity}, confidence={confidence:.2f}"
            )

            return {
                "has_damage": has_damage,
                "damage_type": damage_type,
                "severity": severity,
                "confidence": round(float(confidence), 3),
                "contour_count": len(large_contours),
                "avg_irregularity": round(float(avg_damage_score), 3),
            }

        except Exception as e:
            logger.exception(f"Error in damage detection: {e}")
            return {
                "has_damage": False,
                "damage_type": "error",
                "confidence": 0.0,
                "error": str(e),
            }

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the index and features"""
        total_images = self.index.ntotal
        feature_dimension = self.index.d
        brands = len(self.get_available_brands())

        return {
            "total_images": total_images,
            "feature_dimension": feature_dimension,
            "brands": brands
        }

# Singleton instance
cv_search = None

def get_cv_search() -> CVSearchService:
    """Get or create CV Search service"""
    global cv_search
    if cv_search is None:
        cv_search = CVSearchService()
    return cv_search