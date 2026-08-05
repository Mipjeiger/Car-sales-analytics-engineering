import os
import logging
import joblib
import pandas as pd

# Suppress internal C++ verbosity warnings from XGBoost
os.environ["XGBOOST_VERBOSITY"] = "0"
import xgboost as xgb
from pathlib import Path
from typing import Tuple, Dict, Any
from app.api.models import PredictRequest

logger = logging.getLogger(__name__)

class ModelPredictor:
    def __init__(self):
        # 1. Dynamic Path Resolution for Models Directory
        SCRIPT_DIR = Path(__file__).resolve().parent
        CANDIDATES = [
            Path("/app/development/models"),
            Path("/app/models"),
            SCRIPT_DIR.parents[2] / "models",
            SCRIPT_DIR.parents[3] / "models"
        ]
        
        self.MODELS_DIR = next((p for p in CANDIDATES if p.exists()), CANDIDATES[0])
        logger.info(f"📂 Resolved Predictor Models Directory: {self.MODELS_DIR}")

        self.sales_models: Dict[str, Any] = {}
        self.qty_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.reference_df = self._load_df()

        self._load_models()

    def _load_df(self) -> pd.DataFrame | None:
        """Load a reference DataFrame for feature alignment during prediction."""
        candidates = [
            self.MODELS_DIR.parents[0] / "database" / "car_sales_prediction_sales.parquet",
            self.MODELS_DIR.parents[1] / "database" / "car_sales_prediction_sales.parquet",
            Path(__file__).resolve().parents[3] / "development" / "database" / "car_sales_prediction_sales.parquet",
        ]
        for path in candidates:
            if path.exists():
                logger.info(f"📄 Loaded reference parquet: {path}")
                return pd.read_parquet(path)
            
        logger.warning("⚠️ Reference parquet not found.")
        return None

    def _find_reference_row(self, features: Dict[str, Any]) -> pd.Series | None:
        if self.reference_df is None or self.reference_df.empty:
            logger.warning("⚠️ Reference DataFrame is empty or not loaded.")
            return None

        df = self.reference_df.copy()
        common_keys = [k for k in features.keys() if k in df.columns]

        if common_keys:
            mask = pd.Series(True, index=df.index)
            for col in common_keys:
                mask &= df[col].astype(str) == str(features[col])
            matched = df[mask]
            if not matched.empty:
                logger.info(f"✅ Found reference row for features: {features}")
                return matched.iloc[0]

        return df.iloc[0]  # Fallback to the first row if no match is found

    def _build_features(self, features: Dict[str, Any], source_df: pd.DataFrame | None = None) -> pd.DataFrame:
            """
            Build a Dataframe from the provided features dictionary.
            """
            df = pd.DataFrame([features])

            reference_row = None
            if source_df is not None and not source_df.empty:
                reference_row = source_df.iloc[0]
            else:
                reference_row = self._find_reference_row(features)

            if reference_row is not None:
                for col in reference_row.index:
                    if col not in df.columns:
                        df[col] = reference_row[col]
             
            return df  

    def _load_models(self):
        """Load models and artifact scalers safely."""
        sales_dir = self.MODELS_DIR / "sales_prediction" / "models"
        sales_scalers_dir = self.MODELS_DIR / "sales_prediction" / "scalers"

        qty_dir = self.MODELS_DIR / "quantity_prediction" / "models"
        qty_scalers_dir = self.MODELS_DIR / "quantity_prediction" / "scalers"

        # --- A. Load Sales Models ---
        if sales_dir.exists():
            # 1. XGBoost
            xgb_json = sales_dir / "xgboost.json"
            xgb_pkl = sales_dir / "xgboost.pkl"
            if xgb_json.exists():
                model = xgb.XGBRegressor()
                model.load_model(xgb_json)
                self.sales_models["Xgboost"] = model
            elif xgb_pkl.exists():
                self.sales_models["Xgboost"] = joblib.load(xgb_pkl)

            # 2. Sklearn Models (Random Forest, Decision Tree)
            for m_name in ["random_forest", "decision_tree"]:
                path = sales_dir / f"{m_name}.pkl"
                if path.exists():
                    clean_name = m_name.replace("_", " ").title()
                    self.sales_models[clean_name] = joblib.load(path)

            # 3. CatBoost
            catboost_path = sales_dir / "catboost.cbm"
            if catboost_path.exists():
                try:
                    from catboost import CatBoostRegressor
                    model = CatBoostRegressor()
                    model.load_model(catboost_path)
                    self.sales_models["CatBoost"] = model
                except ImportError:
                    logger.warning("⚠️ CatBoost package not installed. Skipping CatBoost model load.")

        # --- B. Load Scalers ---
        sales_scaler_path = sales_scalers_dir / "feature_scaler.pkl"
        qty_scaler_path = qty_scalers_dir / "feature_scaler.pkl"

        if sales_scaler_path.exists():
            self.scalers["sales"] = joblib.load(sales_scaler_path)
            logger.info(f"✅ Loaded Sales Scaler from {sales_scaler_path}")

        if qty_scaler_path.exists():
            self.scalers["qty"] = joblib.load(qty_scaler_path)
            logger.info(f"✅ Loaded Qty Scaler from {qty_scaler_path}")

    def _encode_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode using labelencoders loaded from the encoders directory."""
        encoders_dir = self.MODELS_DIR / "sales_prediction" / "encoders"

        if not encoders_dir.exists():
            logger.warning(f"⚠️ Encoders directory not found: {encoders_dir}")
            return df

        for col in df.select_dtypes(include=["object"]).columns:
            encoder_path = encoders_dir / f"{col}.pkl"
            if encoder_path.exists():
                encoder = joblib.load(encoder_path)
                df[col] = encoder.transform(df[col].astype(str))
            else:
                logger.warning(f"⚠️ Encoder for column '{col}' not found at {encoder_path}. Skipping encoding.")

        return df

    def predict(self, request: PredictRequest, source_df: pd.DataFrame | None = None) -> Tuple[float, float]:
        """
        Make predictions using selected model.
        Returns a tuple: (prediction_value, confidence_score)
        """
        model_type = request.model_type
        model_name = request.model_name

        # 1. Select active model dictionary
        models = self.sales_models if model_type == "sales" else self.qty_models
        model = models.get(model_name)
        
        if not model:
            available = list(models.keys())
            raise ValueError(
                f"Model '{model_name}' not found for type '{model_type}'. Available: {available}"
            )

        # 2. Convert features dictionary to DataFrame
        features_df = self._build_features(request.features, source_df)

        # 3. Encode string colummns before scaling
        features_df = self._encode_features(features_df)

        # 4. Scale Features and Align Columns
        scaler = self.scalers.get(model_type)
        if scaler:
            expected_cols = list(getattr(scaler, "feature_names_in_", features_df.columns))

            missing_cols = [col for col in expected_cols if col not in features_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required features for scaling: {missing_cols}")

            features_df = features_df.reindex(columns=expected_cols)

            scaled_array = scaler.transform(features_df)
            features_df = pd.DataFrame(scaled_array, columns=expected_cols)

        # 5. Predict
        prediction_val = float(model.predict(features_df)[0])
        confidence_score = 0.95 if prediction_val >= 0 else 0.50

        return prediction_val, confidence_score