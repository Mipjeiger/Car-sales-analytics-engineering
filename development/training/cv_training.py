import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
import numpy as np
import faiss
import json
import pandas as pd
import io
import boto3
from botocore.client import Config
from tqdm import tqdm
import faiss
import os
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Data Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "database" / "Cars_Dataset"

# MinIO Configuration
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
BUCKET_NAME = "mlflow-artifacts"

def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

class CVTrainer:
    """
    Computer Vision Training for Car Images
    Saves FAISS index to MinIO
    """

    def __init__(self):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.s3 = get_minio_client()

        # Ensure bucket exists
        try:
            self.s3.head_bucket(Bucket=BUCKET_NAME)
            logger.info(f"Bucket {BUCKET_NAME} exists.")
        except:
            self.s3.create_bucket(Bucket=BUCKET_NAME)
            logger.info(f"Bucket {BUCKET_NAME} created.")

        logger.info("Loading ViT model for feature extraction...")
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.model.heads = nn.Identity()
        self.model = self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def extract_features(self, image_path):
        """Extract features from single image"""
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze().cpu().numpy()

    def save_to_minio(self, data, key):
        """Save data to MinIO"""
        if isinstance(data, faiss.Index):
            # FAISS index
            import tempfile

            with tempfile.NamedTemporaryFile(suffix=".faiss", delete=False) as tmp:
                faiss.write_index(data, tmp.name)
                with open(tmp.name, "rb") as f:
                    self.s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=f.read())
                os.unlink(tmp.name)

        elif isinstance(data, dict):
            # JSON
            self.s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=json.dumps(data, indent=2))

        elif isinstance(data, pd.DataFrame):
            # CSV
            csv_buffer = io.StringIO()
            data.to_csv(csv_buffer, index=False)
            self.s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=csv_buffer.getvalue())

        elif isinstance(data, np.ndarray):
            # Numpy
            npy_buffer = io.BytesIO()
            np.save(npy_buffer, data)
            self.s3.put_object(Bucket=BUCKET_NAME, Key=key, Body=npy_buffer.getvalue())

        logger.info(f"✅ Saved {key} to MinIO bucket {BUCKET_NAME}")

    def train(self):
        """Extract features and build FAISS index"""
        logger.info("🚀 Starting CV training...")

        image_paths = []
        labels = []
        features = []

        # Process training images
        train_dir = DATA_DIR / "train"
        all_images = []
        for brand in tqdm(train_dir.iterdir(), desc="Processing images"):
            if brand.is_dir():
                all_images.extend([(brand.name, p) for p in brand.glob("*.jpg")])

        if not all_images:
                raise RuntimeError(f"No images found in {train_dir}. Please check the dataset path.")

        progress = tqdm(all_images, desc="Procesing images", disable=not os.isatty(1))

        for brand_name, img_path in progress:
            try:
                feat = self.extract_features(img_path)
                features.append(feat)
                labels.append(brand_name)
                image_paths.append(str(img_path))
            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")

        if not features:
            raise RuntimeError("No features extracted. Please check the dataset and model.")

        # Convert to numpy arrays
        feature_matrix = np.array(features, dtype=np.float32)

        # Normalize features
        norms = np.linalg.norm(feature_matrix, axis=1, keepdims=True)
        feature_matrix = feature_matrix / (norms + 1e-8)

        # Build FAISS index
        dimension = feature_matrix.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(feature_matrix.astype(np.float32))

        # Save to MinIO
        self.save_to_minio(index, "computer_vision/faiss_index.faiss")

        # Save metadata
        metadata = {
            "num_images": len(feature_matrix),
            "feature_dimension": dimension,
            "brands": list(set(labels)),
        }
        self.save_to_minio(metadata, "computer_vision/metadata.json")

        # Save feature data
        df = pd.DataFrame({"path": image_paths, "brand": labels})
        self.save_to_minio(df, "computer_vision/feature_data.csv")

        # Save feature matrix
        self.save_to_minio(feature_matrix, "computer_vision/feature_matrix.npy")

        logger.info("✅ CV training completed and artifacts saved to MinIO.")
        logger.info(f" Images: {len(feature_matrix)}")
        logger.info(f" Brands: {len(set(labels))}")
        logger.info(f" Feature Dimension: {dimension}")


if __name__ == "__main__":
    trainer = CVTrainer()
    trainer.train()
