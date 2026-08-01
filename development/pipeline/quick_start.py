"""
Quick Start: MLflow + DVC Integration with MinIO
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def quick_start():
    print("🚀 Quick Start: MLflow + DVC Integration with MinIO")
    print("="*50)
    
    BASE_DIR = Path.cwd()
    
    # 1. Install dependencies
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'mlflow', 'dvc', 'boto3', 'catboost'])
    
    # 2. Wait for services
    print("⏳ Waiting for services...")
    time.sleep(5)
    
    # 3. Setup DVC with MinIO
    os.chdir(BASE_DIR / 'development' / 'dvc')
    subprocess.run(['dvc', 'init', '--no-scm'])
    subprocess.run(['dvc', 'remote', 'add', 'minio', 's3://mlflow-artifacts'])
    subprocess.run(['dvc', 'remote', 'modify', 'minio', 'endpointurl', 'http://minio:9000'])
    subprocess.run(['dvc', 'remote', 'modify', 'minio', 'access_key_id', 'minioadmin'])
    subprocess.run(['dvc', 'remote', 'modify', 'minio', 'secret_access_key', 'minioadmin'])
    subprocess.run(['dvc', 'remote', 'default', 'minio'])
    
    # 4. Load models to MLflow
    os.chdir(BASE_DIR)
    subprocess.run([sys.executable, str(BASE_DIR / 'development' / 'mlflow' / 'load_existing_models.py')])
    
    # 5. Run integration
    subprocess.run([sys.executable, str(BASE_DIR / 'development' / 'pipeline' / 'dvc_mlflow_integration.py')])
    
    print("\n✅ Quick start complete!")
    print("\nNext steps:")
    print("1. MLflow UI: http://localhost:5003")
    print("2. MinIO Console: http://localhost:9001 (minioadmin/minioadmin)")
    print("3. DVC status: cd development/dvc && dvc status")

if __name__ == "__main__":
    quick_start()