import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Suppress internal C++ verbosity warnings from XGBoost
os.environ["XGBOOST_VERBOSITY"] = "0"
import xgboost as xgb

from app.api.models import PredictRequest

logger = logging.getLogger(__name__)


class ModelPredictor:
    """Service class responsible for loading ML artifacts and executing predictions."""

    def __init__(self) -> None:
        self.models_dir = self._resolve_models_dir()
        logger.info(f"📂 Resolved Predictor Models Directory: {self.models_dir}")

        self.sales_models: Dict[str, Any] = {}
        self.qty_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.encoders: Dict[str, LabelEncoder] = {}

        self.reference_df: Optional[pd.DataFrame] = self._load_reference_dataset()
        self._load_all_artifacts()

    # =========================================================================
    # 1. Path Resolution & Artifact Loading
    # =========================================================================

    def _resolve_models_dir(self) -> Path:
        """Dynamically find the model directory across different deployment environments."""
        script_dir = Path(__file__).resolve().parent
        candidate_paths = [
            Path("/app/development/models"),
            Path("/app/models"),
            script_dir.parents[2] / "models",
            script_dir.parents[3] / "models",
        ]
        return next((path for path in candidate_paths if path.exists()), candidate_paths[0])

    def _load_reference_dataset(self) -> Optional[pd.DataFrame]:
        """Load reference parquet dataset for fallback feature alignment."""
        candidate_paths = [
            self.models_dir.parents[0] / "database" / "car_sales_prediction_sales.parquet",
            self.models_dir.parents[1] / "database" / "car_sales_prediction_sales.parquet",
            Path(__file__).resolve().parents[3] / "development" / "database" / "car_sales_prediction_sales.parquet",
        ]
        for path in candidate_paths:
            if path.exists():
                logger.info(f"📄 Loaded reference parquet from: {path}")
                return pd.read_parquet(path)

        logger.warning("⚠️ Reference parquet dataset not found.")
        return None

    def _load_all_artifacts(self) -> None:
        """Load all models, feature scalers, and label encoders into memory."""
        self._load_sales_models()
        self._load_quantity_models()
        self._load_scalers()
        self._load_encoders()

    def _load_sales_models(self) -> None:
        """Load models for sales prediction."""
        sales_dir = self.models_dir / "sales_prediction" / "models"
        if not sales_dir.exists():
            return

        # XGBoost Model
        xgb_json = sales_dir / "xgboost.json"
        xgb_pkl = sales_dir / "xgboost.pkl"
        if xgb_json.exists():
            model = xgb.XGBRegressor()
            model.load_model(xgb_json)
            self.sales_models["Xgboost"] = model
        elif xgb_pkl.exists():
            self.sales_models["Xgboost"] = joblib.load(xgb_pkl)

        # Scikit-Learn Models
        for model_key in ["random_forest", "decision_tree"]:
            model_path = sales_dir / f"{model_key}.pkl"
            if model_path.exists():
                display_name = model_key.replace("_", " ").title()
                self.sales_models[display_name] = joblib.load(model_path)

        # CatBoost Model
        catboost_path = sales_dir / "catboost.cbm"
        if catboost_path.exists():
            try:
                from catboost import CatBoostRegressor
                model = CatBoostRegressor()
                model.load_model(catboost_path)
                self.sales_models["CatBoost"] = model
            except ImportError:
                logger.warning("⚠️ CatBoost is not installed. Skipping CatBoost model load.")

    def _load_quantity_models(self) -> None:
        """Load models for quantity prediction."""
        qty_dir = self.models_dir / "quantity_prediction" / "models"
        if not qty_dir.exists():
            return

        for model_file in qty_dir.glob("*.pkl"):
            model_name = model_file.stem.replace("_", " ").title()
            self.qty_models[model_name] = joblib.load(model_file)

    def _load_scalers(self) -> None:
        """Load feature scalers for sales and quantity domains."""
        sales_scaler_path = self.models_dir / "sales_prediction" / "scalers" / "feature_scaler.pkl"
        qty_scaler_path = self.models_dir / "quantity_prediction" / "scalers" / "feature_scaler.pkl"

        if sales_scaler_path.exists():
            self.scalers["sales"] = joblib.load(sales_scaler_path)
            logger.info(f"✅ Loaded Sales Scaler from {sales_scaler_path}")

        if qty_scaler_path.exists():
            self.scalers["qty"] = joblib.load(qty_scaler_path)
            logger.info(f"✅ Loaded Quantity Scaler from {qty_scaler_path}")

    def _load_encoders(self) -> None:
        """Unpack nested categorical label encoders from single artifact dictionary."""
        encoder_path = self.models_dir / "sales_prediction" / "encoders" / "label_encoders.pkl"
        if not encoder_path.exists():
            logger.warning(f"⚠️ Label encoders artifact not found at {encoder_path}")
            return

        raw_encoders: Dict[str, Any] = joblib.load(encoder_path)
        for col_name, encoder_obj in raw_encoders.items():
            # Handle nested structure: {'col_name': {0: LabelEncoder()}}
            if isinstance(encoder_obj, dict) and 0 in encoder_obj:
                self.encoders[col_name] = encoder_obj[0]
            elif isinstance(encoder_obj, LabelEncoder):
                self.encoders[col_name] = encoder_obj

        logger.info(f"✅ Successfully loaded {len(self.encoders)} categorical label encoders.")

    # =========================================================================
    # 2. Feature Engineering & Preprocessing
    # =========================================================================

    def _find_reference_row(self, input_features: Dict[str, Any]) -> Optional[pd.Series]:
        """Find a baseline matching row from reference dataset for missing input features."""
        if self.reference_df is None or self.reference_df.empty:
            logger.warning("⚠️ Reference DataFrame is unavailable for feature imputation.")
            return None

        df = self.reference_df.copy()
        matching_columns = [col for col in input_features if col in df.columns]

        if matching_columns:
            mask = pd.Series(True, index=df.index)
            for col in matching_columns:
                mask &= (df[col].astype(str) == str(input_features[col]))
            matched_rows = df[mask]
            
            if not matched_rows.empty:
                logger.info("✅ Found matching reference row for missing feature imputation.")
                return matched_rows.iloc[0]

        return df.iloc[0]

    def _build_feature_dataframe(
        self, input_features: Dict[str, Any], source_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Construct complete single-row DataFrame using inputs and reference values."""
        feature_df = pd.DataFrame([input_features])

        reference_row = None
        if source_df is not None and not source_df.empty:
            reference_row = source_df.iloc[0]
        else:
            reference_row = self._find_reference_row(input_features)

        if reference_row is not None:
            for col in reference_row.index:
                if col not in feature_df.columns:
                    feature_df[col] = reference_row[col]

        return feature_df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns with unknown-class fallback."""
        if not self.encoders:
            logger.warning("⚠️ No encoders available. Skipping categorical encoding.")
            return df

        for col in df.columns:
            if col not in self.encoders:
                continue

            encoder = self.encoders[col]
            series_str = df[col].astype(str)

            try:
                df[col] = encoder.transform(series_str)
            except ValueError:
                # Handle unknown out-of-vocabulary categorical values during inference
                known_classes = set(encoder.classes_)
                fallback_class = encoder.classes_[0]
                cleaned_series = series_str.apply(lambda x: x if x in known_classes else fallback_class)
                
                df[col] = encoder.transform(cleaned_series)
                logger.warning(
                    f"⚠️ Unseen value in column '{col}'. Fallback applied using '{fallback_class}'."
                )

        return df

    def _scale_and_align_features(
        self, df: pd.DataFrame, domain_type: str
    ) -> pd.DataFrame:
        """Align columns to match training schema and apply feature scaling."""
        scaler = self.scalers.get(domain_type)
        if not scaler:
            return df

        expected_columns = list(getattr(scaler, "feature_names_in_", df.columns))
        missing_columns = [col for col in expected_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required features for scaling: {missing_columns}")

        df = df.reindex(columns=expected_columns)
        scaled_array = scaler.transform(df)
        
        return pd.DataFrame(scaled_array, columns=expected_columns)

    # =========================================================================
    # 3. Main Inference Pipeline
    # =========================================================================

    def predict(
        self, request: PredictRequest, source_df: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float]:
        """
        Execute prediction pipeline for incoming request.

        Args:
            request: The model inference request containing model metadata and feature dict.
            source_df: Optional DataFrame to override baseline reference features.

        Returns:
            Tuple[float, float]: (prediction_value, confidence_score)
        """
        model_type = request.model_type
        model_name = request.model_name

        # 1. Model Resolution
        available_models = self.sales_models if model_type == "sales" else self.qty_models
        selected_model = available_models.get(model_name)

        if not selected_model:
            raise ValueError(
                f"Model '{model_name}' not found for domain '{model_type}'. "
                f"Available models: {list(available_models.keys())}"
            )

        # 2. Pipeline Transformations
        features_df = self._build_feature_dataframe(request.features, source_df)
        features_df = self._encode_categorical_features(features_df)
        features_df = self._scale_and_align_features(features_df, model_type)

        # 3. Inference & Post-processing
        raw_prediction = float(selected_model.predict(features_df)[0])
        confidence_score = 0.95 if raw_prediction >= 0 else 0.50

        return raw_prediction, confidence_score