import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import joblib
import numpy as np
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

    # ===========================================================================
    # Model inspection methods
    # ===========================================================================
    def get_available_models(self) -> Dict[str, List[str]]:
        return {
            "sales_models": list(self.sales_models.keys()),
            "quantity_models": list(self.qty_models.keys()),
        }

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
            model = xgb.Booster()
            model.load_model(xgb_json)
            self.sales_models["Xgboost"] = model

        elif xgb_pkl.exists():
            loaded_model = joblib.load(xgb_pkl)

            # 🛠️ Patch missing legacy XGBoost attributes
            if not hasattr(loaded_model, "gpu_id"):
                setattr(loaded_model, "gpu_id", None)
            if not hasattr(loaded_model, "predictor"):
                setattr(loaded_model, "predictor", None)

            self.sales_models["Xgboost"] = loaded_model

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
            self.scalers["quantity"] = joblib.load(qty_scaler_path)
            logger.info(f"✅ Loaded Quantity Scaler from {qty_scaler_path}")

    def _load_encoders(self) -> None:
        """Unpack nested categorical label encoders from single artifact dictionary."""
        encoder_path = self.models_dir / "sales_prediction" / "encoders" / "label_encoders.pkl"
        if not encoder_path.exists():
            logger.warning(f"⚠️ Label encoders artifact not found at {encoder_path}")
            return

        raw_encoders: Dict[str, Any] = joblib.load(encoder_path)
        for col_name, encoder_obj in raw_encoders.items():
            if isinstance(encoder_obj, dict):
                # Search inside dictionary values for the actual LabelEncoder instance
                for val in encoder_obj.values():
                    if isinstance(val, LabelEncoder):
                        self.encoders[col_name] = val
                        break # Stop after finding the first LabelEncoder instance

            elif isinstance(encoder_obj, LabelEncoder):
                self.encoders[col_name] = encoder_obj

        logger.info(f"✅ Successfully loaded {len(self.encoders)} categorical label encoders.")

    # =========================================================================
    # 2. Feature Engineering & Preprocessing
    # =========================================================================

    def _find_reference_row(self, input_features: Dict[str, Any]) -> Optional[pd.Series]:
        """Find a baseline matching row from reference dataset for missing input features."""
        if self.reference_df is None or self.reference_df.empty:
            return None

        df = self.reference_df.copy()

        # Filtered reference dataset by categorical matches in input features 
        if "model" in input_features and "model" in df.columns:
            matched = df[df["model"].astype(str).str.lower() == str(input_features["model"]).lower()]
            if not matched.empty:
                return matched.select_dtypes(include=[np.number]).median()

    def _build_feature_dataframe(
        self, input_features: Dict[str, Any], source_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Construct complete single-row DataFrame using inputs and reference values."""
        feature_dict = input_features.copy()

        # Compute basic derived features dynamically
        if "profit" not in feature_dict and "gross_sales" in feature_dict and "cost" in feature_dict:
            feature_dict["profit"] = float(feature_dict["gross_sales"] - feature_dict["cost"])

        # Default date features if missing
        now = pd.Timestamp.now()
        feature_dict.setdefault("day_of_week", now.dayofweek)
        feature_dict.setdefault("week_of_year", now.isocalendar().week)
        feature_dict.setdefault("season", (now.month % 12 + 3) // 3)  # 1=Winter, 2=Spring, 3=Summer, 4=Fall

        feature_df = pd.DataFrame([feature_dict])

        # Fill remaining missing feature from aggregated reference row if available
        reference_row = source_df.iloc[0] if (source_df is not None and not source_df.empty) else self._find_reference_row(feature_dict)

        if reference_row is not None:
            for col in feature_df.columns:
                if col not in feature_df.columns:
                    feature_df[col] = reference_row[col]

        return feature_df

    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply label encoding to categorical columns with unknown-class fallback."""
        if not self.encoders:
            return df

        for col in df.columns:
            if col not in self.encoders:
                continue

            encoder = self.encoders[col]

            if not hasattr(encoder, "classes_"):
                logger.warning(f"⚠️ Encoder for column '{col}' is missing 'classes_' attribute. Skipping encoding.")
                continue

            series_str = df[col].astype(str)

            try:
                df[col] = encoder.transform(series_str)
            except ValueError:
                # Handle unknown out-of-vocabulary categorical values during inference
                known_classes = set(encoder.classes_.tolist())
                fallback_class = encoder.classes_[0]

                # Map unknown categorical values to the fallback class
                cleaned_series = series_str.apply(lambda x: x if x in known_classes else fallback_class)
                df[col] = encoder.transform(cleaned_series)

                logger.warning(f"⚠️ Unseen value in column '{col}'. Fallback applied using '{fallback_class}'.")

        return df

    def _scale_and_align_features(
        self, df: pd.DataFrame, domain_type: str, selected_model: Any
    ) -> pd.DataFrame:
        """Align columns to match training schema and apply feature scaling."""
        # Convert numeric features safely
        for col in df.columns:
            if col not in self.encoders:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        expected_columns: List[str] = []

        # 1. Determine target feature list
        scaler = self.scalers.get(domain_type)
        if isinstance(selected_model, xgb.Booster):
            expected_columns = selected_model.feature_names or []
        elif hasattr(selected_model, "feature_names_in_"):
            expected_columns = list(selected_model.feature_names_in_)
        elif scaler and hasattr(scaler, "feature_names_in_"):
            expected_columns = list(scaler.feature_names_in_)

        # 2. Reindex to align feature columns
        if expected_columns:
            missing = [c for c in expected_columns if c not in df.columns]
            if missing:
                logger.warning(f"⚠️ Missing expected features for domain '{domain_type}': {missing}. Filling with zeros.")
                for m_col in missing:
                    df[m_col] = 0

            # Drop any extra columns not in expected list
            df = df[expected_columns]
            logger.info(f"📊 Final Dataframe shape before scaling: {df.shape}")

        # 3. Check tree model status 🛠️ [FIX: Removed invalid outer any()]
        model_class_name = selected_model.__class__.__name__.lower()
        is_tree_model = isinstance(selected_model, xgb.Booster) or any(
            tree_type in model_class_name
            for tree_type in ["randomforest", "decisiontree", "xgb", "catboost"]
        )

        if scaler and not is_tree_model:
            logger.info(f"📊 Applying feature scaler for non-tree model: {model_class_name}")
            scaled_array = scaler.transform(df)
            return pd.DataFrame(scaled_array, columns=df.columns)

        logger.info(f"⚡ Skipping feature scaling for tree-based model: {model_class_name}")
        return df
        
    # =========================================================================
    # 3. Main Inference Pipeline
    # =========================================================================

    def _calculate_confidence_score(
        self, selected_model: Any, features_df: pd.DataFrame, raw_prediction: float
    ) -> float:
        """Calculate realistic confidence score for regression models"""

        # Ensure all features are numeric for confidence calculations
        try:
            numeric_df = features_df.copy()
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

            logger.info(f"📊 Confidence calc features dtype:\n{numeric_df.dtypes}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to convert features to numeric for confidence calculation: {e}")
            return 0.50 

        # --- Approach 1: Ensemble Variance (For Random Forest / Decision Trees) ---
        if hasattr(selected_model, "estimators_"):
            try:
                # Get prediction from all individual decision trees
                tree_predictions = np.array([
                    tree.predict(features_df.values)[0] for tree in selected_model.estimators_
                ])

                std_dev = np.std(tree_predictions)
                mean_pred = np.mean(tree_predictions)

                if mean_pred == 0:
                    return 0.50

                # Coficient of Variation (CV = std_dev / mean)
                cv = abs(std_dev / mean_pred)
                confidence = float(np.exp(-cv))
                return round(float(np.clip(confidence, 0.10, 0.99)), 3)

            except Exception as e:
                logger.warning(f"⚠️ Could not calculate tree ensemble confidence: {e}")

        # --- Approach 2: Prediction Magnitude & Residual Fallback (XGBoost/CatBoost) ---
        # For gradient boosted trees or single trees, use relative magnitude heuristics
        if raw_prediction < 0:
            return 0.15

        # Assuming predictions closer to typical ranges hold higher
        dampened_score = 1.0 / (1.0 + np.exp(-raw_prediction / (abs(raw_prediction) + 1e-5)))
        return round(float(np.clip(dampened_score * 0.90, 0.35, 0.95)), 3)

    def _predict_single_domain(
        self, domain_key: str, model_name: str, feature_dict: Dict[str, Any], source_df: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float]:
        available_models = self.qty_models if domain_key == "qty" else self.sales_models
        selected_model = available_models.get(model_name)

        if not selected_model:
            raise ValueError(
                f"Model '{model_name}' not found for domain '{domain_key}'. "
                f"Available models: {list(available_models.keys())}"
            )

        # Preprocessing & Alignment
        features_df = self._build_feature_dataframe(feature_dict, source_df)
        logger.info(f"🔍 [DEBUG] Initial features_df before encoding:\n{features_df.to_dict()}")

        features_df_encoded = self._encode_categorical_features(features_df.copy())
        logger.info(f"🔍 [DEBUG] Features after encoding:\n{features_df_encoded.to_dict()}")

        features_df_scaled = self._scale_and_align_features(features_df_encoded.copy(), domain_key, selected_model)
        logger.info(f"🔍 [DEBUG] Features after scaling and alignment:\n{features_df_scaled.to_dict()}")

        # 🛠️ [FIX 2]: Ensure gpu_id exists on selected_model before inference
        if not hasattr(selected_model, "gpu_id"):
            setattr(selected_model, "gpu_id", None)
        if not hasattr(selected_model, "predictor"):
            setattr(selected_model, "predictor", None)

        # Inference
        if isinstance(selected_model, xgb.Booster):
            dmatrix = xgb.DMatrix(features_df_scaled)
            raw_pred = float(selected_model.predict(dmatrix)[0])
        else:
            raw_pred = float(selected_model.predict(features_df_scaled)[0])

        logger.info(f"📊 Raw prediction from model: {raw_pred}")

        final_pred = self._scale_prediction_to_range(raw_pred, domain_key)
        confidence = self._calculate_confidence_score(selected_model, features_df_scaled, raw_pred)
        logger.info(f"📈 Confidence score for prediction: {confidence}")

        return final_pred, confidence

    def _scale_prediction_to_range(self, raw_pred: float, domain_key: str) -> float:
        """
        Return raw prediction directly if model was trained on unscaled targets.
        Only denormalize if target standardization was explicitly applied during training.
        """
        scaler = self.scalers.get(domain_key)
        
        if domain_key == "sales":
            target_col = "sales"
            
            # 🛠️ [FIX 1]: Use scaler's mean and std to denormalize z-score
            if scaler and hasattr(scaler, "mean_"):
                # Get the index of the target column in the scaler
                feature_names = list(scaler.feature_names_in_)
                
                if target_col in feature_names:
                    target_idx = feature_names.index(target_col)
                    
                    mean = float(scaler.mean_[target_idx])
                    scale = float(scaler.scale_[target_idx])  # or std for StandardScaler
                    
                    # Denormalize: original = z_score * scale + mean
                    denormalized = (raw_pred * scale) + mean
                    
                    logger.info(f"📊 Denormalizing sales z-score {raw_pred}:")
                    logger.info(f"   Mean: {mean}, Scale: {scale}")
                    logger.info(f"   Denormalized: {denormalized}")
                    
                    return denormalized
            
            # 🛠️ [FIX 2]: Fallback to reference dataset if scaler unavailable
            if self.reference_df is None or self.reference_df.empty:
                logger.warning("⚠️ Reference DataFrame and scaler unavailable. Returning raw prediction.")
                return raw_pred

            min_val = float(self.reference_df[target_col].min())
            max_val = float(self.reference_df[target_col].max())
            mean_val = float(self.reference_df[target_col].mean())
            std_val = float(self.reference_df[target_col].std())

            logger.info(f"📊 Reference sales stats - Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Std: {std_val}")
            logger.info(f"📊 Raw prediction (z-score): {raw_pred}")

            # Denormalize using reference stats
            denormalized = (raw_pred * std_val) + mean_val
            clipped = np.clip(denormalized, min_val * 0.8, max_val * 1.2)
            
            logger.info(f"🔄 Denormalized to {denormalized}, clipped to {clipped}")
            return clipped

        elif domain_key == "qty":
            target_col = "quantity"
            
            # 🛠️ [FIX 1]: Use scaler's mean and std to denormalize z-score
            if scaler and hasattr(scaler, "mean_"):
                feature_names = list(scaler.feature_names_in_)
                
                if target_col in feature_names:
                    target_idx = feature_names.index(target_col)
                    
                    mean = float(scaler.mean_[target_idx])
                    scale = float(scaler.scale_[target_idx])
                    
                    # Denormalize
                    denormalized = (raw_pred * scale) + mean
                    denormalized = max(1, int(round(denormalized)))
                    
                    logger.info(f"📊 Denormalizing quantity z-score {raw_pred}:")
                    logger.info(f"   Mean: {mean}, Scale: {scale}")
                    logger.info(f"   Denormalized: {denormalized}")
                    
                    return denormalized
            
            # 🛠️ [FIX 2]: Fallback to reference dataset
            if self.reference_df is None or self.reference_df.empty:
                logger.warning("⚠️ Reference DataFrame and scaler unavailable. Returning raw prediction.")
                return max(1, int(round(raw_pred)))

            min_val = float(self.reference_df[target_col].min())
            max_val = float(self.reference_df[target_col].max())
            mean_val = float(self.reference_df[target_col].mean())
            std_val = float(self.reference_df[target_col].std())

            # Denormalize using reference stats
            denormalized = (raw_pred * std_val) + mean_val
            denormalized = max(1, int(round(np.clip(denormalized, min_val, max_val))))
            
            logger.info(f"🔄 Denormalized quantity to {denormalized}")
            return denormalized

        return raw_pred

    def predict(
        self, request: PredictRequest, source_df: Optional[pd.DataFrame] = None
    ) -> Tuple[float, float]:
        """Execute dual or single prediction based on request."""

        # Normalize model domain key
        raw_type = request.model_type.lower()
        sales_m_name = request.sales_model_name or request.model_name
        qty_m_name = request.quantity_model_name or request.model_name

        # Wrap all prediction results in a dictionary
        response_data: Dict[str, Any] = {
            "model_type": raw_type,
            "predicted_sales": None,
            "sales_confidence": None,
            "predicted_quantity": None,
            "quantity_confidence": None,
        }

        # Predict sales (if requested or if "both")
        if raw_type in ["sales", "both", "all"]:
            raw_sales, sales_conf = self._predict_single_domain(
                "sales", sales_m_name, request.features, source_df
            )
            response_data["predicted_sales"] = round(max(0.0, raw_sales), 2)
            response_data["sales_confidence"] = sales_conf

        # Predict quantity (if requested or if "both")
        if raw_type in ["quantity", "both", "all"]:
            raw_qty, qty_conf = self._predict_single_domain(
                "qty", qty_m_name, request.features, source_df
            )
            response_data["predicted_quantity"] = max(1, int(round(raw_qty)))
            response_data["quantity_confidence"] = qty_conf

        return response_data