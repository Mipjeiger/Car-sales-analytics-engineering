import logging
import pandas as pd
import numpy as np
import torch
import json
import torch.nn as nn
import faiss
import logging
from pathlib import Path
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from typing import Optional

# Loggger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration database
PATH_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PATH_DIR / "models" / "quality_assurance" / "database" / "clean_image_dataset.csv"
IMAGE_ROOT = PATH_DIR / "database" / "cars_damage_dataset"

# 1. Load dataset .parquet path
def load_data() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        return None
    
    df = pd.read_csv(DATA_PATH)

    logger.info(
        f"Data loaded successfully | "
        f"rows={len(df):,} | "
        f"columns={len(df.columns)}"
    )
    logger.info(f"Data preview:\n{df.head()}")
    return df

# 2. Load extension images (.jpg, .png, .jpeg)
def resolve_image_path(image_path: str) -> Optional[Path]:
    if not image_path or pd.isna(image_path):
        return None

    image_path = str(image_path).strip()

    # case 1: Path already exists
    path = Path(image_path)
    if path.is_absolute() and path.exists():
        return path

    # Case 2: Image_path contains
    marker = "cars_damage_dataset"
    if marker in image_path:
        relative_part = image_path.split(marker, 1)[1]
        relative_part = relative_part.lstrip("/\\")
        resolved_path = IMAGE_ROOT / relative_part
        return resolved_path

    # Case 3: Image path is something like image/0.jpeg, image/1.jpeg, etc.
    resolved_path = IMAGE_ROOT / image_path
    return resolved_path

def load_images(image_paths: list[str]) -> list[Image.Image]:
    images = []
    try:
        for image_path in image_paths:
            resolved_path = resolve_image_path(image_path)

            if resolved_path is None:
                logger.warning(f"Image path is None or invalid: {image_path}")
                continue

            if not resolved_path.exists():
                logger.warning(f"Image file not found: {resolved_path}")
                continue

            image = Image.open(resolved_path).convert("RGB")
            images.append(image)

    except Exception as e:
        logger.error(f"Error loading images: {e}")
        return None

    return images

# Define class for training computer vision model for car damage identification
class CarDamageModelTrainer:
    def __init__(self, model_name: str = "google/vit-base-patch16-224", device: Optional[str] = None):
        self.device = device or {"mps" if torch.backends.mps.is_available() else "cpu"}
        self.model_name = model_name
        self.model = ViTForImageClassification.from_pretrained(model_name).to(self.device)
        self.processor = ViTImageProcessor.from_pretrained(model_name)

# 3. Load severity_mapping.json
# 4. Setup Vision Transformer model
# 5. Embedding the images and severity mapping to the model
# 6. Train loop epochs with early stopping the model on the dataset
# 7. Build feature inspection_id, car_id, inspection_date, image_path, predicted_damage, confidence, severity, qa_status, inspector, 
# repair_cost, repair_days, processing_time_ms, customer_name, dealer_name, company, model, discount
# 8. Build Faiss Index for similarity search based on the predicted damage and severity mapping
# 9. Implemented recommendation car system (Similarity search) based on the predicted damage and severity mapping