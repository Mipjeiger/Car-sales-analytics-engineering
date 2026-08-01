#!/bin/bash
# ============================================================
# MLflow + DVC + MinIO Integration Script
# Run from: Car_Sales/development/pipeline/
# ============================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# Configuration
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEV_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$DEV_DIR")"
DOCKER_COMPOSE_FILE="$DEV_DIR/docker-compose.yml"

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}🚀 MLflow + DVC + MinIO Integration Pipeline${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${YELLOW}Script Dir:${NC} $SCRIPT_DIR"
echo -e "${YELLOW}Dev Dir:${NC} $DEV_DIR"
echo -e "${YELLOW}Project Root:${NC} $PROJECT_ROOT"

MLFLOW_PORT=5003
MINIO_PORT=9000
MINIO_CONSOLE_PORT=9001

# ============================================================
# Helper Functions
# ============================================================
check_service() {
    local service=$1
    local port=$2
    local health_check=$3
    
    echo -ne "${YELLOW}Checking $service...${NC}"
    if curl -s "http://localhost:$port$health_check" > /dev/null 2>&1; then
        echo -e " ${GREEN}✅ Running${NC}"
        return 0
    else
        echo -e " ${RED}❌ Not responding${NC}"
        return 1
    fi
}

wait_for_service() {
    local service=$1
    local port=$2
    local health_check=$3
    local max_attempts=30
    local attempt=0
    
    echo -ne "${YELLOW}Waiting for $service...${NC}"
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "http://localhost:$port$health_check" > /dev/null 2>&1; then
            echo -e " ${GREEN}✅ Ready${NC}"
            return 0
        fi
        attempt=$((attempt + 1))
        echo -n "."
        sleep 2
    done
    
    echo -e " ${RED}❌ Timeout${NC}"
    return 1
}

# ============================================================
# Step 1: Start Docker Services
# ============================================================
echo -e "\n${BLUE}[Step 1] Starting Docker Services...${NC}"

if [ ! -f "$DOCKER_COMPOSE_FILE" ]; then
    echo -e "${RED}❌ docker-compose.yml not found at: $DOCKER_COMPOSE_FILE${NC}"
    exit 1
fi

# Detect docker-compose CLI format
DOCKER_CMD=""
if command -v docker &> /dev/null && docker compose version &> /dev/null; then
    DOCKER_CMD="docker compose -f $DOCKER_COMPOSE_FILE"
elif command -v docker-compose &> /dev/null; then
    DOCKER_CMD="docker-compose -f $DOCKER_COMPOSE_FILE"
else
    echo -e "${RED}❌ Docker / Docker Compose not found. Please install Docker.${NC}"
    exit 1
fi

# Start services using explicit compose file
$DOCKER_CMD up -d

echo -e "${YELLOW}Waiting for services to initialize...${NC}"
sleep 3

# Check PostgreSQL
if ! $DOCKER_CMD exec -T postgres pg_isready -U car_analytics_user > /dev/null 2>&1; then
    echo -e "${YELLOW}⏳ Waiting for PostgreSQL...${NC}"
    sleep 3
fi

# Check MinIO & MLflow
wait_for_service "MinIO" 9000 "/minio/health/ready"
wait_for_service "MLflow" 5003 "/health"

echo -e "${GREEN}✅ All Docker services started${NC}"

# ============================================================
# Step 2: Install Python Dependencies (with progress output)
# ============================================================
echo -e "\n${BLUE}[Step 2] Installing Python Dependencies...${NC}"

cd "$DEV_DIR" || exit 1

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --progress-bar on
else
    pip install mlflow dvc dvc-s3 boto3 catboost pandas numpy scikit-learn joblib --progress-bar on
fi

echo -e "${GREEN}✅ Dependencies installed${NC}"

# ============================================================
# Step 3: Create MinIO Bucket
# ============================================================
echo -e "\n${BLUE}[Step 3] Creating MinIO Bucket...${NC}"

python -c "
import boto3
from botocore.client import Config
import time

time.sleep(2)

s3 = boto3.client(
    's3',
    endpoint_url='http://localhost:9000',
    aws_access_key_id='minioadmin',
    aws_secret_access_key='minioadmin',
    config=Config(signature_version='s3v4'),
    region_name='us-east-1'
)

try:
    s3.head_bucket(Bucket='mlflow-artifacts')
    print('✅ Bucket already exists: mlflow-artifacts')
except:
    s3.create_bucket(Bucket='mlflow-artifacts')
    print('✅ Bucket created: mlflow-artifacts')
" 2>/dev/null || echo -e "${YELLOW}⚠️ Unable to create bucket (may already exist)${NC}"

# ============================================================
# Step 4: Load Models to MLflow
# ============================================================
echo -e "\n${BLUE}[Step 4] Loading Models to MLflow...${NC}"

cd "$DEV_DIR" || exit 1

if [ -f "mlflow/load_existing_models.py" ]; then
    python mlflow/load_existing_models.py
else
    echo -e "${RED}❌ mlflow/load_existing_models.py not found${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Models loaded to MLflow${NC}"

# ============================================================
# Step 5: Setup DVC
# ============================================================
echo -e "\n${BLUE}[Step 5] Setting up DVC with MinIO...${NC}"

mkdir -p "$DEV_DIR/dvc"
cd "$DEV_DIR/dvc" || exit 1

if [ ! -d ".dvc" ]; then
    echo -e "${YELLOW}Initializing DVC...${NC}"
    dvc init --no-scm --quiet 2>/dev/null || dvc init --no-scm
fi

if ! dvc remote list | grep -q "minio"; then
    echo -e "${YELLOW}Adding MinIO remote...${NC}"
    dvc remote add -d minio s3://mlflow-artifacts
    dvc remote modify minio endpointurl http://localhost:9000
    dvc remote modify minio access_key_id minioadmin
    dvc remote modify minio secret_access_key minioadmin
else
    echo -e "${GREEN}✅ MinIO remote already configured${NC}"
fi

echo -e "\n${YELLOW}DVC Remote Configuration:${NC}"
dvc remote list

# ============================================================
# Step 6: Run MLflow + DVC Integration
# ============================================================
echo -e "\n${BLUE}[Step 6] Running MLflow + DVC Integration...${NC}"

# 1. First, generate dvc.yaml and model_versions.json
cd "$DEV_DIR/pipeline" || exit 1

if [ -f "dvc_mlflow_integration.py" ]; then
    echo -e "${YELLOW}Generating DVC configuration files...${NC}"
    python dvc_mlflow_integration.py
else
    echo -e "${RED}❌ dvc_mlflow_integration.py not found${NC}"
    exit 1
fi

# 2. Next, switch to development/dvc where dvc.yaml lives and execute pipeline
cd "$DEV_DIR/dvc" || exit 1

echo -e "${YELLOW}Running DVC pipeline (dvc repro)...${NC}"
dvc repro

echo -e "${GREEN}✅ Integration complete${NC}"

# ============================================================
# Step 7: Check DVC Status & Tracked Files
# ============================================================
echo -e "\n${BLUE}[Step 7] Checking DVC Status & Metrics...${NC}"

cd "$DEV_DIR/dvc" || exit 1

echo -e "${YELLOW}DVC Status:${NC}"
dvc status

echo -e "${YELLOW}Tracked Files in DVC:${NC}"
# Pass '.' as the url target for the current repository
dvc list . --dvc-only

echo -e "${YELLOW}DVC Metrics:${NC}"
dvc metrics show

# ============================================================
# Step 8: Test Model Serving
# ============================================================
echo -e "\n${BLUE}[Step 8] Testing Model Serving...${NC}"

cd "$DEV_DIR/mlflow" || exit 1

if [ -f "serve_models.py" ]; then
    python -c "
import mlflow
import json
mlflow.set_tracking_uri('http://localhost:5003')
client = mlflow.tracking.MlflowClient()
models = client.search_registered_models()
print(f'✅ Found {len(models)} registered models:')
for m in models:
    latest = client.get_latest_versions(m.name, stages=['Production'])
    version = latest[0].version if latest else 'N/A'
    print(f'  - {m.name}: v{version}')
"
else
    echo -e "${YELLOW}⚠️ serve_models.py not found, skipping...${NC}"
fi

# ============================================================
# Step 9: Summary
# ============================================================
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ Integration Complete!${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e ""
echo -e "${YELLOW}📊 Services:${NC}"
echo -e "  MLflow UI:     ${GREEN}http://localhost:$MLFLOW_PORT${NC}"
echo -e "  MinIO Console: ${GREEN}http://localhost:$MINIO_CONSOLE_PORT${NC}"
echo -e "  MinIO API:     ${GREEN}http://localhost:$MINIO_PORT${NC}"
echo -e ""
echo -e "${YELLOW}🔑 Credentials:${NC}"
echo -e "  MinIO:         ${GREEN}minioadmin / minioadmin${NC}"
echo -e ""
echo -e "${YELLOW}📁 Useful Commands (from development dir):${NC}"
echo -e "  Check DVC:     ${BLUE}cd dvc && dvc status${NC}"
echo -e "  Show Metrics:  ${BLUE}cd dvc && dvc metrics show${NC}"
echo -e "  Run Pipeline:  ${BLUE}cd dvc && dvc repro${NC}"
echo -e "  MLflow UI:     ${BLUE}open http://localhost:$MLFLOW_PORT${NC}"
echo -e ""
echo -e "${BLUE}============================================================${NC}"

cd "$SCRIPT_DIR" || exit 1