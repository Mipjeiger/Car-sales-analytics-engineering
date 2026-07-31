"""
==========================
MLFLOW + DVC Integration FOR Model Management
==========================
"""

import mlflow
import subprocess
import json
import pandas as pd
from pathlib import Path
import shutil
import yaml

class MLflowDVCIntegration:
    def __init__(self):
        self.BASE_DIR = Path.cwd()
        self.MLFLOW_DIR = self.BASE_DIR / 'development' / 'mlflow'
        self.DVC_DIR = self.BASE_DIR / 'development' / 'dvc'
        self.MODELS_DIR = self.BASE_DIR / 'development' / 'models'

        # Setup MLflow tracking URI
        mlflow.set_tracking_uri(f"sqlite:///{self.MLFLOW_DIR / 'mlflow.db'}")