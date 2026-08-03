import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import mlflow
from dagster import job, op

# Resolve paths dynamically
SCRIPT_DIR = Path(__file__).resolve().parent  # development/orchestration/pipelines
DEV_DIR = SCRIPT_DIR.parents[2]                # development
MODELS_DIR = DEV_DIR / "models"
DATABASE_DIR = DEV_DIR / "database"

# MLflow Tracking URI
mlflow.set_tracking_uri("http://localhost:5003")

def _preprocess_data(df: pd.DataFrame, target_dir: Path, target_type: str):
    """Applies saved Encoders and Scalers to raw dataframe."""
    df_processed = df.copy()

    # 1. Load Feature Columns
    feature_file = target_dir / "parameters" / "feature_columns.json"
    if feature_file.exists():
        with open(feature_file, "r") as f:
            feature_cols = json.load(f)
    else:
        feature_cols = [c for c in df.columns if c not in ['sales', 'quantity']]

    # 2. Apply Label Encoders if present
    encoders_file = target_dir / "encoders" / "label_encoders.pkl"
    if encoders_file.exists():
        encoders = joblib.load(encoders_file)
        for col, encoder in encoders.items():
            if col in df_processed.columns:
                df_processed[col] = encoder.transform(df_processed[col].astype(str))

    # Keep only target features
    X = df_processed[[col for col in feature_cols if col in df_processed.columns]]
    y = df['sales'] if target_type == "sales" else df['quantity']

    # 3. Apply Feature Scalers if present
    scaler_file = target_dir / "scalers" / "feature_scaler.pkl"
    if scaler_file.exists():
        scaler = joblib.load(scaler_file)
        X_scaled = scaler.transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    return X, y, target_dir / "scalers" / "target_scaler.pkl"

def _evaluate_all_dir_models(df: pd.DataFrame, target_type: str):
    """Loads and evaluates ALL models found in target models/ directory."""
    target_dir = MODELS_DIR / ("sales_prediction" if target_type == "sales" else "quantity_prediction")
    models_folder = target_dir / "models"

    X, y_true, target_scaler_path = _preprocess_data(df, target_dir, target_type)
    target_scaler = joblib.load(target_scaler_path) if target_scaler_path.exists() else None

    # Discover all model files in models/ folder
    model_files = list(models_folder.glob("*.*"))
    results = {}

    mlflow.set_experiment(f"dagster_{target_type}_all_models")

    for model_file in model_files:
        algo_name = model_file.stem
        ext = model_file.suffix.lower()

        # Skip non-model files if any
        if ext not in ['.cbm', '.pkl', '.json']:
            continue

        try:
            # Load Model based on extension/type
            if ext == '.cbm' or 'catboost' in algo_name:
                model = CatBoostRegressor()
                model.load_model(str(model_file))

            elif ext == '.json' and 'xgboost' in algo_name:
                model = XGBRegressor()
                model.load_model(str(model_file))
                
            elif ext == '.pkl':
                # Skip if it's an extra xgboost pkl when json is already evaluated
                if 'xgboost' in algo_name and (models_folder / "xgboost.json").exists() and ext == '.pkl':
                    continue
                model = joblib.load(model_file)
            else:
                continue

            # Predict
            y_pred = model.predict(X)

            # Inverse scale target predictions if target_scaler exists
            if target_scaler:
                y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()

            r2 = float(r2_score(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mape = float(mean_absolute_percentage_error(y_true, y_pred))

            results[algo_name] = {"r2": r2, "rmse": rmse, "mape": mape}

            # Log each model to MLflow
            with mlflow.start_run(run_name=f"{target_type}_{algo_name}"):
                mlflow.log_params({"target": target_type, "algorithm": algo_name})
                mlflow.log_metrics({"r2": r2, "rmse": rmse, "mape": mape})

        except Exception as e:
            results[algo_name] = {"error": str(e)}

    return results

@op
def load_data():
    """Load sales dataset."""
    parquet_path = DATABASE_DIR / "car_sales_prediction_sales.parquet"
    if not parquet_path.exists():
        parquet_path = DATABASE_DIR / "car_sales.parquet"
    return pd.read_parquet(parquet_path)

@op
def evaluate_all_sales_models(df: pd.DataFrame):
    """Evaluates ALL models (CatBoost, XGBoost, Random Forest, Decision Tree) in sales_prediction."""
    return _evaluate_all_dir_models(df, target_type="sales")

@op
def evaluate_all_quantity_models(df: pd.DataFrame):
    """Evaluates ALL models (CatBoost, XGBoost, Random Forest, Decision Tree) in quantity_prediction."""
    return _evaluate_all_dir_models(df, target_type="quantity")

@op
def summarize_all_results(sales_results: dict, quantity_results: dict):
    """Combines evaluation metrics for all models across both targets."""
    return {
        "sales_models_evaluation": sales_results,
        "quantity_models_evaluation": quantity_results
    }

@job
def training_pipeline():
    """Dagster Job that evaluates ALL models in both target directories."""
    df = load_data()
    
    sales_res = evaluate_all_sales_models(df)
    quantity_res = evaluate_all_quantity_models(df)
    
    summarize_all_results(sales_res, quantity_res)