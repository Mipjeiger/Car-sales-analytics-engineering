import io
import json
import logging
import os
import pickle
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import boto3
import faiss
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from botocore.client import Config as BotoConfig
from dotenv import load_dotenv
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification, ViTImageProcessor

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration & Path Resolution
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model_name: str = "google/vit-base-patch16-224"
    batch_size: int = 32
    epochs: int = 2
    lr: float = 2e-5
    
    project_root: Path = field(default_factory=lambda: Config._find_root())
    image_root: Path = field(init=False)
    data_path: Path = field(init=False)
    model_dir: Path = field(init=False)
    
    # MinIO / S3 Setup
    minio_endpoint: str = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
    minio_access_key: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
    minio_secret_key: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
    bucket_name: str = os.getenv("MLFLOW_ARTIFACT_BUCKET", "mlflow-artifacts")

    def __post_init__(self):
        # Load environment variables
        env_file = self.project_root / "env" / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        self.model_dir = self.project_root / "models" / "quality_assurance"
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.data_path = self.model_dir / "database" / "clean_image_dataset.csv"
        self.image_root = self._resolve_image_root()

    @staticmethod
    def _find_root() -> Path:
        search_paths = [
            Path("/app"),
            Path("/opt/airflow"),
            Path(__file__).resolve().parents[2],
            Path(__file__).resolve().parents[1],
        ]
        for path in search_paths:
            if any((path / target).exists() for target in ["database", "models", "development"]):
                return path
        return Path(__file__).resolve().parents[2]

    def _resolve_image_root(self) -> Path:
        candidates = [
            self.project_root / "database" / "cars_damage_dataset",
            Path("/app/database/cars_damage_dataset"),
            Path("/opt/airflow/database/cars_damage_dataset"),
        ]
        return next((p for p in candidates if p.exists()), candidates[0])

cfg = Config()

def get_device() -> torch.device:
    """Select best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

# ---------------------------------------------------------------------------
# Storage Client (MinIO / S3)
# ---------------------------------------------------------------------------
class MinIOHandler:
    def __init__(self, config: Config = cfg):
        self.config = config
        self.client = boto3.client(
            "s3",
            endpoint_url=config.minio_endpoint,
            aws_access_key_id=config.minio_access_key,
            aws_secret_access_key=config.minio_secret_key,
            config=BotoConfig(signature_version="s3v4"),
            region_name="us-east-1",
        )

    def save(self, data: any, key: str) -> None:
        if isinstance(data, dict):
            body = json.dumps(data, indent=2).encode("utf-8")
        elif isinstance(data, pd.DataFrame):
            body = data.to_csv(index=False).encode("utf-8")
        elif isinstance(data, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, data)
            body = buf.getvalue()
        elif isinstance(data, bytes):
            body = data
        else:
            buf = io.BytesIO()
            pickle.dump(data, buf)
            body = buf.getvalue()

        self.client.put_object(Bucket=self.config.bucket_name, Key=key, Body=body)
        logger.info(f"Saved artifact to MinIO: {key}")

    def load(self, key: str) -> Optional[bytes]:
        try:
            response = self.client.get_object(Bucket=self.config.bucket_name, Key=key)
            return response["Body"].read()
        except Exception as e:
            logger.error(f"Error fetching {key} from MinIO: {e}")
            return None

minio_handler = MinIOHandler()

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------
def resolve_image_path(image_path: str, config: Config = cfg) -> Optional[Path]:
    """Deterministically resolve relative or absolute image path."""
    if not image_path or pd.isna(image_path):
        return None

    path_str = str(image_path).strip()

    # 1. Direct hit check
    direct_path = Path(path_str)
    if direct_path.exists() and direct_path.is_file():
        return direct_path

    # 2. Strip relative traversal operators
    clean_path_str = path_str.replace("../", "").lstrip("/")

    # 3. Handle 'cars_damage_dataset' subpath matching
    if "cars_damage_dataset" in clean_path_str:
        sub_path = clean_path_str.split("cars_damage_dataset")[-1]
        candidate = config.image_root / sub_path
        if candidate.exists() and candidate.is_file():
            return candidate

    # 4. Fallback to image root
    filename = Path(path_str).name
    candidates = [
        config.project_root / clean_path_str,
        config.image_root / filename,
        config.image_root / "image" / filename,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    return None

def load_dataset(config: Config = cfg) -> Optional[pd.DataFrame]:
    """Load clean dataset CSV and prepare image path links."""
    path = config.data_path if config.data_path.exists() else None
    if not path:
        logger.error("Data file path does not exist.")
        return None

    df = pd.read_csv(path)
    if "image" in df.columns:
        df["image_path"] = df["image"].apply(lambda x: str(resolve_image_path(x) or ""))

        # Drop rows where images could not be resolved
        missing_count = (df["image_path"] == "").sum()
        if missing_count > 0:
            logger.warning(f"{missing_count} images could not be resolved and will be dropped.")
            df = df[df["image_path"] != ""].reset_index(drop=True)

    return df

def get_severity_mapping() -> Dict:
    """Retrieve or initialize severity & repair cost mappings."""
    mapping = {
        "severity": {
            "no_damage": {"severity": 0, "level": "None", "repair_days": 0},
            "01-minor": {"severity": 1, "level": "Low", "repair_days": 1},
            "02-moderate": {"severity": 2, "level": "Medium", "repair_days": 3},
            "03-severe": {"severity": 3, "level": "High", "repair_days": 7},
        },
        "cost": {
            "no_damage": 0,
            "01-minor": 300_000,
            "02-moderate": 1_000_000,
            "03-severe": 3_000_000,
        },
    }
    return mapping

# ---------------------------------------------------------------------------
# PyTorch Dataset Definition
# ---------------------------------------------------------------------------
class CarDamageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor: ViTImageProcessor, image_col: str = "image_path", label_col: str = "classes"):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.image_col = image_col

        self.label_encoder = LabelEncoder()
        labels = self.df[label_col].fillna("no_damage").astype(str)
        self.df["encoded_label"] = self.label_encoder.fit_transform(labels)
        self.num_classes = len(self.label_encoder.classes_)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.df.iloc[idx][self.image_col]
        label = self.df.iloc[idx]["encoded_label"]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as err:
            logger.warning(f"Failed loading image at index {idx}: {err}")
            image = Image.new("RGB", (224, 224), "gray")

        inputs = self.processor(images=image, return_tensors="pt")
        return inputs["pixel_values"].squeeze(0), torch.tensor(label, dtype=torch.long)

# ---------------------------------------------------------------------------
# ViT Model Trainer & Evaluator
# ---------------------------------------------------------------------------
class CarDamageTrainer:
    def __init__(self, config: Config = cfg):
        self.config = config
        self.device = get_device()
        self.processor = ViTImageProcessor.from_pretrained(config.model_name)
        self.label_encoder: Optional[LabelEncoder] = None
        self.model: Optional[ViTForImageClassification] = None

    def initialize_model(self, num_labels: int):
        self.model = ViTForImageClassification.from_pretrained(
            self.config.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)

    def prepare_dataloaders(self, df: pd.DataFrame) -> Tuple[DataLoader, DataLoader]:
        train_df, val_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df.get("classes")
        )
        train_ds = CarDamageDataset(train_df, self.processor)
        val_ds = CarDamageDataset(val_df, self.processor)

        self.label_encoder = train_ds.label_encoder
        self.initialize_model(train_ds.num_classes)

        train_loader = DataLoader(train_ds, 
                                  batch_size=self.config.batch_size, 
                                  shuffle=True, 
                                  num_workers=0, 
                                  persistent_workers=False)
        
        val_loader = DataLoader(val_ds, 
                                batch_size=self.config.batch_size, 
                                shuffle=False, 
                                num_workers=0, 
                                persistent_workers=False)
        
        return train_loader, val_loader

    def train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for pixels, labels in tqdm(loader, desc="Training", leave=False):
            pixels, labels = pixels.to(self.device), labels.to(self.device)

            optimizer.zero_grad()

            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = self.model(pixel_values=pixels)
                loss = criterion(outputs.logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(loader), correct / total

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct, total = 0, 0

        with torch.no_grad():
            for pixels, labels in tqdm(loader, desc="Evaluating", leave=False):
                pixels, labels = pixels.to(self.device), labels.to(self.device)
                outputs = self.model(pixel_values=pixels)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def train(self, df: pd.DataFrame) -> List[Dict]:
        train_loader, val_loader = self.prepare_dataloaders(df)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        history = []

        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_acc = self.evaluate(val_loader)

            logger.info(f"Epoch {epoch+1}/{self.config.epochs} - Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})

            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model()

        return history

    def save_model(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            torch.save(self.model.state_dict(), tmp_path / "model.pth")
            self.processor.save_pretrained(tmp_path)
            
            if self.label_encoder:
                with open(tmp_path / "label_encoder.pkl", "wb") as f:
                    pickle.dump(self.label_encoder, f)

            for file in tmp_path.iterdir():
                minio_handler.save(file.read_bytes(), f"quality_assurance/{file.name}")

        logger.info("Successfully exported trained model artifacts to MinIO.")

# ---------------------------------------------------------------------------
# Vector Search Pipeline (FAISS)
# ---------------------------------------------------------------------------
class CarDamageSearch:
    def __init__(self, trainer: CarDamageTrainer):
        self.trainer = trainer
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: Optional[pd.DataFrame] = None

    def build_index(self, df: pd.DataFrame):
        self.trainer.model.eval()
        features, metadata = [], []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Vector Features"):
            resolved = resolve_image_path(row.get("image_path", ""))
            if resolved:
                img = Image.open(resolved).convert("RGB")
                inputs = self.trainer.processor(images=img, return_tensors="pt")
                pixels = inputs["pixel_values"].to(self.trainer.device)

                with torch.no_grad():
                    logits = self.trainer.model(pixel_values=pixels).logits.cpu().numpy().squeeze()
                    features.append(logits)
                    metadata.append({"image_path": str(resolved), "classes": row.get("classes", "unknown")})

        if not features:
            logger.warning("No valid images found for FAISS indexing.")
            return

        feat_arr = np.array(features)
        norms = np.linalg.norm(feat_arr, axis=1, keepdims=True)
        normalized = (feat_arr / (norms + 1e-8)).astype(np.float32)

        self.index = faiss.IndexFlatIP(normalized.shape[1])
        self.index.add(normalized)
        self.metadata = pd.DataFrame(metadata)

        # Export Index to MinIO
        with tempfile.NamedTemporaryFile(suffix=".faiss") as tmp:
            faiss.write_index(self.index, tmp.name)
            minio_handler.save(Path(tmp.name).read_bytes(), "quality_assurance/damage_index.faiss")

        minio_handler.save(self.metadata, "quality_assurance/damage_metadata.csv")
        logger.info(f"FAISS index with {self.index.ntotal} items built and stored.")

# ---------------------------------------------------------------------------
# Execution Entry Point
# ---------------------------------------------------------------------------
def run_training():
    logger.info("🚀 Initiating Car Damage Pipeline Training...")
    df = load_dataset()
    if df is None or df.empty:
        logger.error("Dataset empty or unavailable. Aborting execution.")
        return None, None, []

    trainer = CarDamageTrainer()
    history = trainer.train(df)

    searcher = CarDamageSearch(trainer)
    searcher.build_index(df)

    return trainer, searcher, history

if __name__ == "__main__":
    trainer, searcher, history = run_training()