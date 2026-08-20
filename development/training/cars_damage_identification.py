import logging
import pandas as pd
import numpy as np
import torch
import json
import torch.nn as nn
import torch.optim as optim
import faiss
import logging
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
from typing import Optional, Dict, List, Tuple
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
import pickle
from tqdm import tqdm
import os
import io
import boto3
from botocore.client import Config
from datetime import datetime
import tempfile

# Loggger configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration database
# Path configuration
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "models" / "quality_assurance" / "database" / "clean_image_dataset.csv"
IMAGE_ROOT = BASE_DIR / "database" / "cars_damage_dataset"
MODEL_DIR = BASE_DIR / "models" / "quality_assurance"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables
ENV_DIR = BASE_DIR / "env"
if ENV_DIR.exists():
    load_dotenv(ENV_DIR / ".env")

# MinIO configuration
MINIO_ENDPOINT = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://minio:9000')
MINIO_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID')
MINIO_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
BUCKET_NAME = os.getenv('MLFLOW_ARTIFACT_BUCKET', 'mlflow-artifacts')

# ============================================================
# MinIO Client Setup
# ===========================================================
def get_minio_client():
    return boto3.client(
        's3',
        endpoint_url=MINIO_ENDPOINT,
        aws_access_key_id=MINIO_ACCESS_KEY,
        aws_secret_access_key=MINIO_SECRET_KEY,
        config=Config(signature_version='s3v4'),
        region_name='us-east-1'
    )

def save_to_minio(data, key: str, bucket: str = BUCKET_NAME):
    """Save data to MinIO"""
    s3 = get_minio_client()
    
    if isinstance(data, dict):
        data = json.dumps(data, indent=2)
    elif isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
    elif isinstance(data, np.ndarray):
        buffer = io.BytesIO()
        np.save(buffer, data)
        data = buffer.getvalue()
    elif isinstance(data, bytes):
        pass
    else:
        # Try to pickle
        buffer = io.BytesIO()
        pickle.dump(data, buffer)
        data = buffer.getvalue()
    
    s3.put_object(Bucket=bucket, Key=key, Body=data)
    logger.info(f"✅ Saved to MinIO: {key}")

def load_from_minio(key: str, bucket: str = BUCKET_NAME):
    """Load data from MinIO"""
    s3 = get_minio_client()
    try:
        response = s3.get_object(Bucket=bucket, Key=key)
        return response['Body'].read()
    except Exception as e:
        logger.error(f"❌ Error loading from MinIO: {e}")
        return None

# ============================================================
# 1. Load dataset .parquet path
# ==========================================================

def load_data() -> pd.DataFrame | None:
    if not DATA_PATH.exists():
        logger.error(f"Data file not found: {DATA_PATH}")
        return None
    
    df = pd.read_csv(DATA_PATH)
    logger.info(f"Data loaded: rows={len(df):,}, columns={len(df.columns)}")

    return df
    
# ============================================================
# 2. Load extension images (.jpg, .png, .jpeg)
# =========================================================

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
    for image_path in image_paths:
            resolved_path = resolve_image_path(image_path)
            if resolved_path and resolved_path.exists():
                try:

                    image = Image.open(resolved_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    logger.error(f"Error loading image {resolved_path}: {e}")

    return images

# ============================================================
# 3. Dataset Class for Car Damage Identification
# =========================================================

class CarDamageDataset(Dataset):
    """Dataset class for car damage identification using Vision Transformer"""
    def __init__(self, df: pd.DataFrame, processor, image_col: str = 'image_path'):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_col = image_col

        # Encode labels
        self.label_encoder = LabelEncoder()
        self.df['damage_encoded'] = self.label_encoder.fit_transform(self.df['damage_type'].fillna('no_damage'))
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row[self.image_col]

        resolved_path = resolve_image_path(image_path)
        if resolved_path and resolved_path.exists():
            image = Image.open(resolved_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color='gray')

        # Process image
        inputs = self.processor(image=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze()

        # Get label
        label = row['damage_encoded']

        return pixel_values, label

# ===========================================================
# 4. Severity & Cost Mapping
# ===========================================================
def load_severity_mapping() -> Dict:
    """Load severity mapping from JSON
        Returns:
            Create default dictionary severity mapping JSON"""
    severity_path = MODEL_DIR / "severity_mapping_build.json"

    default_mapping = {
    "no_damage": {"severity": 0, "level": "None", "repair_days": 0},
    "bumper_scrape": {"severity": 1, "level": "Low", "repair_days": 1},
    "door_scratch": {"severity": 1, "level": "Low", "repair_days": 1},
    "dent": {"severity": 2, "level": "Medium", "repair_days": 3},
    "head_lamp": {"severity": 2, "level": "Medium", "repair_days": 3},
    "broken_headlight": {"severity": 3, "level": "High", "repair_days": 7}
    }

    cost_mapping = {
    "no_damage": 0,
    "bumper_scrape": 500_000,
    "door_scratch": 750_000,
    "dent": 1_200_000,
    "head_lamp": 2_500_000,
    "broken_headlight": 4_500_000
    }

    # Save default mapping to JSON
    full_mapping = {"severity": default_mapping, "cost": cost_mapping}
    with open(severity_path, "w") as f:
        json.dump(full_mapping, f, indent=2)

    return full_mapping

def estimate_repair_cost(damage_class: str, confidence: float) -> Dict:
    """Estimate repair cost based on damage class and confidence"""
    cost_mapping = load_severity_mapping()['cost']
    base_cost = cost_mapping.get(damage_class, 0)

    # Adjust cost based on confidence
    adjusted_cost = base_cost * (1 + (1 - confidence) * 0.2)
    final_cost = int(round(adjusted_cost, -3))

    return {
        'cost_idr': final_cost,
        'formatted_cost': f"Rp {final_cost:,.0f}".replace(",", "."),
        'confidence_score': round(confidence, 2),
        'confidence_percentage': f"{confidence * 100:.2f}%"
    }

def estimate_repair_days(severity_level: str) -> int:
    days_mapping = {
        'None': 0,
        'Low': 1,
        'Medium': 3,
        'High': 7
    }
    return days_mapping.get(severity_level, 1)

# ===========================================================
# 5. Vision Transformer Model Training
# ===========================================================

class CarDamageTrainer:
    """Trainer class for Vision Transformer model for car damage identification"""
    def __init__(self, model_name: str = "google/vit-base-patch16-224"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model_name = model_name

        # Load model and processor
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=6,
            ignore_mismatched_sizes=True
        ).to(self.device)

        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.seveity_mapping = load_severity_mapping()
        self.label_encoder = None

        logger.info(f"✅ Model and processor loaded: {model_name} on device {self.device}")

    def prepare_dataloaders(self, df: pd.DataFrame, batch_size: int = 16, test_size: float = 0.2):
        """Prepare train and validation dataloaders"""
        # Split data
        train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)

        # Create datasets
        train_dataset = CarDamageDataset(train_df, self.processor)
        val_dataset = CarDamageDataset(val_df, self.processor)

        # Store label encoder
        self.label_encoder = train_dataset.label_encoder
        self.num_classes = train_dataset.num_classes

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        logger.info(f"✅ Dataloaders prepared: train={len(train_dataset)}, val={len(val_dataset)}")
        return train_loader, val_loader

    def train_epoch(self, train_loader, optimizer, criterion):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for pixel_values, labels in tqdm(train_loader, desc="Training", leave=False):
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)

            # Optimization step
            optimizer.zero_grad()
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Update metrics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(train_loader)
        return avg_loss, accuracy

    def evaluate(self, val_loader):
        """Evaluate model on validation set"""
        self.model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_confidences = []

        with torch.no_grad():
            for pixel_values, labels in tqdm(val_loader, desc="Evaluating"):
                pixel_values = pixel_values.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(pixel_values=pixel_values)
                probs = torch.softmax(outputs.logits, dim=1)
                confidence, predicted = torch.max(probs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_confidences.extend(confidence.cpu().numpy())

            accuracy = correct / total
            return accuracy, all_preds, all_labels, all_confidences

    def train(self, df: pd.DataFrame, epochs: int = 10, batch_size: int = 16, lr: float = 2e-5):
        """Train. model with early stopping"""
        train_loader, val_loader = self.prepare_dataloaders(df, batch_size=batch_size)

        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0
        patience = 3
        patience_counter = 0
        training_history = []

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_acc, _, _, _ = self.evaluate(val_loader)

            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            })

            logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model() # Save best model
                logger.info(f"✅ New best model saved with Val Acc: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info("Early stopping triggered.")
                    break

        return training_history

    def save_model(self):
        """Save model to MinIO"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model weights
            model_path = Path(tmpdir) / "model.pth"
            torch.save(self.model.state_dict(), model_path)
            
            # Save processor config
            self.processor.save_pretrained(tmpdir)
            
            # Save label encoder
            if self.label_encoder:
                with open(Path(tmpdir) / "label_encoder.pkl", 'wb') as f:
                    pickle.dump(self.label_encoder, f)
            
            # Upload all files to MinIO
            for file in Path(tmpdir).iterdir():
                with open(file, 'rb') as f:
                    key = f"quality_assurance/{file.name}"
                    save_to_minio(f.read(), key)
        
        # Save severity mapping
        save_to_minio(self.severity_mapping, "quality_assurance/severity_mapping.json")
        
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'num_classes': self.num_classes,
            'classes': self.label_encoder.classes_.tolist() if self.label_encoder else [],
            'timestamp': datetime.now().isoformat()
        }
        save_to_minio(metadata, "quality_assurance/metadata.json")
        
        logger.info("✅ Model saved to MinIO")

    def load_model_from_minio(self):
        """Load model from MinIO"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Download files
            for file_name in ['model.pth', 'config.json', 'preprocessor_config.json', 'label_encoder.pkl']:
                data = load_from_minio(f"quality_assurance/{file_name}")
                if data:
                    with open(Path(tmpdir) / file_name, 'wb') as f:
                        f.write(data)

            try:
                # Load model weights
                self.model.load_state_dict(
                    torch.load(Path(tmpdir) / "model.pth", map_location=self.device)
                )
                self.model.to(self.device)
                self.model.eval()
                
                # Load processor
                self.processor = ViTImageProcessor.from_pretrained(tmpdir)
                
                # Load label encoder
                with open(Path(tmpdir) / "label_encoder.pkl", 'rb') as f:
                    self.label_encoder = pickle.load(f)
                
                # Load severity mapping
                severity_data = load_from_minio("quality_assurance/severity_mapping.json")
                if severity_data:
                    self.severity_mapping = json.loads(severity_data)
                
                logger.info("✅ Model loaded from MinIO")
            
            except:
                # Fallback using default model and processor
                self.model = ViTForImageClassification.from_pretrained(
                    self.model_name,
                    num_labels=6,
                    ignore_mismatched_sizes=True
                ).to(self.device)
                self.processor = ViTImageProcessor.from_pretrained(self.model_name)
                logger.warning("⚠️ Failed to load model from MinIO. Using default model and processor.")

# ============================================================
# 6. Inference Pipeline
# ============================================================

def inference_pipeline(
        image_path: str,
        model,
        processor,
        label_encoder,
        severity_mapping,
        threshold: float = 0.7,
        device: str = "mps"
) -> Dict:
    """Run inference on a single image or batch of images"""
    start_time = datetime.now()

    try:
        # Load and process image
        img = Image.open(image_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(device)

        # Inference
        model.eval()
        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            probs = torch.softmax(outputs.logits, dim=1)
            confidence, pred = torch.max(probs, 1)

        pred_class = label_encoder.classes_[pred.item()]
        confidence_score = confidence.item()

        # Get severity
        severity_info = severity_mapping['severity'].get(pred_class, {'level': 'Low'})
        severity_level = severity_info.get('level', 'Low')

        # Estimate repair cost
        repair_cost = estimate_repair_cost(pred_class, confidence_score)
        repair_days = estimate_repair_days(severity_level)

        # QA Status
        if confidence_score >= threshold:
            qa_status = "Pass" if repair_days <= 3 else "Rework"
        else:
            qa_status = "Fail"

        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        return {
            'image_path': str(image_path),
            'predicted_damage': pred_class,
            'confidence': confidence_score,
            'confidence_percentage': f"{confidence_score * 100:.2f}%",
            'severity': severity_level,
            'qa_status': qa_status,
            'repair_cost': repair_cost['cost_idr'],
            'repair_cost_formatted': repair_cost['formatted_cost'],
            'repair_days': repair_days,
            'processing_time_ms': round(processing_time, 2),
            'success': True
        }

    except Exception as e:
        logger.error(f"Error during inference: {e}")
        return {
            'image_path': str(image_path),
            'error': str(e),
            'success': False
        }

# ===========================================================
# 7. Faiss Index Builder
# ===========================================================

class CarDamageSearch:
    def __init__(self, trainer: CarDamageTrainer):
        self.trainer = trainer
        self.index = None
        self.features = None
        self.metadata = None

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features from images"""
        self.trainer.model.eval()
        features = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Features"):
            image_path = row['image_path']
            resolved_path = resolve_image_path(image_path)

            if resolved_path and resolved_path.exists():
                image = Image.open(resolved_path).convert("RGB")
                inputs = self.trainer.processor(images=image, return_tensors="pt")
                pixel_values = inputs['pixel_values'].to(self.trainer.device)

                with torch.no_grad():
                    outputs = self.trainer.model(pixel_values=pixel_values)
                    feature = outputs.logits.cpu().numpy().squeeze()
                    features.append(feature)

            else:
                features.append(np.zeros(self.trainer.num_classes))

        return np.array(features)

    def build_index(self, df: pd.DataFrame):
        """Build FAISS index from similarity search"""
        self.features = self.extract_features(df)
        self.metadata = df[['inspection_id', 'car_id', 'damage_type']].copy()

        # Normalize features
        norms = np.linalg.norm(self.features, axis=1, keepdims=True)
        normalized_features = self.features / (norms + 1e-8)

        # Build FAISS index
        dimension = normalized_features.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(normalized_features.astype(np.float32))

        # Save index to MinIO
        with tempfile.TemporaryDirectory(suffix='.faiss', delete=False) as tmp:
            faiss.write_index(self.index, tmp.name)
            with open(tmp.name, 'rb') as f:
                save_to_minio(f.read(), "quality_assurance/damage_index.faiss")
            os.unlink(tmp.name)

        # Save metadata
        save_to_minio(self.metadata, "quality_assurance/damage_metadata.csv")
        
        logger.info(f"✅ FAISS index built: {self.index.ntotal} vectors")

    def search(self, query_image: Image.Image, k: int = 5) -> List[Dict]:
        """Search for similar damage cases"""
        if self.index is None:
            logger.error("FAISS index not built. Please build the index first.")
            return []

        # Extract query features
        inputs = self.trainer.processor(images=query_image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.trainer.device)

        with torch.no_grad():
            outputs = self.trainer.model(pixel_values=pixel_values)
            query_feature = outputs.logits.cpu().numpy().squeeze()

        # Normalize
        query_features = query_features / (np.linalg.norm(query_features) + 1e-8)
        query_features = query_features.reshape(1, -1).astype(np.float32)

        # Search
        distances, indices = self.index.search(query_features, k)

        # Get results
        results = []

        for idx, score in zip(indices[0], distances[0]):
            if idx < len(self.metadata):
                row = self.metadata.iloc[idx]
                results.append({
                    'inspection_id': row.get('inspection_id', 'N/A'),
                    'car_id': row.get('car_id', 'N/A'),
                    'damage_type': row.get('damage_type', 'Unknown'),
                    'similarity': float(score)
                })

        return results

# ===========================================================
# 8. Main Training Function
# ===========================================================

def run_training():
    """Run complete training pipeline for car damage identification"""
    logger.info("🚀 Starting Car Damage Identification Training")

    # 1. Load data
    df = load_data()
    if df is None:
        logger.error("❌ Data loading failed. Exiting training.")
        return None, None

    # 2. Initialize trainer
    trainer = CarDamageTrainer()

    # 3. Train model
    history = trainer.train(df, epochs=10, batch_size=16)

    # 4. Build FAISS index
    searcher = CarDamageSearch(trainer)
    searcher.build_index(df)

    logger.info("✅ Training and index building completed.")
    return trainer, searcher, history

def run_inference_on_image(image_path: str):
    """Run inference on a single image using trained model"""
    trainer = CarDamageTrainer()
    trainer.load_model_from_minio()

    # Run inference
    result = inference_pipeline(
        image_path=image_path,
        model=trainer.model,
        processor=trainer.processor,
        label_encoder=trainer.label_encoder,
        severity_mapping=trainer.seveity_mapping,
        device=trainer.device
    )

    return result

if __name__ == "__main__":

    # Example usage
    trainer, searcher, history = run_training()
    test_image_path = IMAGE_ROOT / "image" / "20.jpeg"
    inference_result = run_inference_on_image(test_image_path)
    logger.info(f"Inference Result: {inference_result}")
    logger.info(f"Training History: {history}")
    logger.info("✅ Car Damage Identification Training and Inference Completed")