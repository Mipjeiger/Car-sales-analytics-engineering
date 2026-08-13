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
import os
import logging

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Data Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'database' / 'Cars_Dataset'

# MinIO Configuration
MINIO_ENDPOINT = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('MINIO_ACCESS_KEY')
MINIO_SECRET_KEY = os.getenv('MINIO_SECRET_KEY')
BUCKET_NAME = 'mlflow-artifacts'

def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
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

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, image_path):
        """Extract features from single image"""
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze().cpu().numpy()