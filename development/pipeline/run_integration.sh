#!/bin/bash
echo "🚀 Starting MLflow + DVC Integration with MinIO..."
echo "==================================================="

# Check MinIO
echo "⏳ Checking MinIO..."
curl -s http://localhost:9000/minio/health/ready > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ MinIO is running"
else
    echo "⚠️ Starting MinIO..."
    cd ../../
    docker-compose up -d minio
    sleep 5
    cd development/pipeline
fi

# Load models to MLflow
echo "📊 Loading models to MLflow..."
cd ../mlflow
python3 load_existing_models.py
cd ../pipeline

# Setup DVC
echo "📦 Setting up DVC..."
cd ../dvc
dvc init --no-scm
dvc remote add -d minio s3://mlflow-artifacts
dvc remote modify minio endpointurl http://minio:9000
dvc remote modify minio access_key_id minioadmin
dvc remote modify minio secret_access_key minioadmin
cd ../pipeline

# Run integration
echo "🔄 Running integration..."
python3 dvc_mlflow_integration.py

# Show status
echo "📊 DVC Status:"
cd ../dvc
dvc status
dvc metrics show

echo ""
echo "✅ Integration complete!"
echo "📊 MLflow UI: http://localhost:5003"
echo "🪣 MinIO Console: http://localhost:9001"