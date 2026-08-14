#!/usr/bin/env python3
"""
Generate a clean DVC pipeline for MLflow model loading and evaluation.

Rules:
- load models from development/models
- store DVC data in development/dvc
- do not track mlflow.db in DVC
- do not list metrics/evaluation.json in outs and metrics at the same time
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from dotenv import load_dotenv


class MLflowDVCIntegration:
    def __init__(self) -> None:
        self.pipeline_dir = Path(__file__).resolve().parent
        self.dev_dir = self.pipeline_dir.parent
        self.root_dir = self.dev_dir.parent
        self.dvc_dir = self.dev_dir / "dvc"

        # Check dir path first, fallback to root models
        self.models_dir = self.dev_dir / "models"
        if not self.models_dir.exists():
            self.models_dir = self.root_dir / "models"

        self.mlflow_dir = self.dev_dir / "mlflow"

        load_dotenv(self.dev_dir / ".env")

        # Load envrionment variables
        env_path = self.dev_dir / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
            print(f"🔑 Loaded environment variables from: {env_path}")

        self.dvc_dir.mkdir(parents=True, exist_ok=True)
        (self.dvc_dir / "data").mkdir(parents=True, exist_ok=True)
        (self.dvc_dir / "metrics").mkdir(parents=True, exist_ok=True)

    def _get_cv_dir(self) -> Path:
        """Resolve Computer vision directory path dynamically"""
        cv_dir = self.models_dir / "computer_vision_2"
        if not cv_dir.exists():
            cv_dir = self.root_dir / "computer_vision"
        return cv_dir

    def sync_models_manifest(self) -> None:
        """
        Create a small manifest of local model folders.
        This uses development/models only.
        """
        manifest = {}
        model_paths = [
            self.models_dir / "sales_prediction" / "models",
            self.models_dir / "quantity_prediction" / "models",
            self._get_cv_dir(),
        ]

        for model_path in model_paths:
            try:
                rel_path = str(model_path.relative_to(self.dev_dir))
            except ValueError:
                rel_path = str(model_path.relative_to(self.root_dir))

            manifest[rel_path] = {
                "exists": model_path.exists(),
                "type": "dir" if model_path.is_dir() else "missing",
            }

        target = self.dvc_dir / "models_versions.json"
        target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"✅ Updated models manifest: {target}")

    def generate_dvc_yaml(self) -> None:
        """Generate DVC pipeline file with relative dependencies and outputs"""
        cv_folder_name = self._get_cv_dir().name
        config = {
            "stages": {
                "load_mlflow_models": {
                    "cmd": "MODELS_DIR=../models DVC_DIR=. python3 ../pipeline/stages/load_models.py",
                    "deps": [
                        "../models/sales_prediction/models/",
                        "../models/quantity_prediction/models/",
                        f"../models/{cv_folder_name}/",
                    ],
                    "outs": ["data/"],
                },
                "evaluate_models": {
                    "cmd": "DVC_DIR=. python3 ../pipeline/stages/evaluate.py",
                    "deps": ["data/"],
                    "metrics": [
                        {"metrics/evaluation.json": {"cache": False}},
                    ],
                },
            }
        }

        target = self.dvc_dir / "dvc.yaml"
        target.write_text(
            yaml.safe_dump(config, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
        )
        print(f"Updated: {target}")

    def run(self) -> None:
        print("🚀 Generating DVC integration files...")
        self.sync_models_manifest()
        self.generate_dvc_yaml()
        print("✨ Done generating configuration!")


if __name__ == "__main__":
    MLflowDVCIntegration().run()
