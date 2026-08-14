from app.api.models import PredictRequest
import xgboost as xgb
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


logger = logging.getLogger(__name__)


class ModelPredictor:
    """Service class responsible for loading ML artifacts and executing predictions."""

    def __init__(self) -> None:
        self.models_dir = self._resolve_models_dir()
        logger.info(f"📂 Resolved Predictor Models Directory: {self.models_dir}")

        self.sales_models: Dict[str, Any] = {}
        self.qty_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.target_scalers: Dict[str, Any] = {}
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
            Path(__file__).resolve().parents[3]
            / "development"
            / "database"
            / "car_sales_prediction_sales.parquet",
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

        xgb_json = sales_dir / "xgboost.json"
        xgb_pkl = sales_dir / "xgboost.pkl"

        if xgb_json.exists():
            model = xgb.Booster()
            model.load_model(xgb_json)
            self.sales_models["Xgboost"] = model

        elif xgb_pkl.exists():
            loaded_model = joblib.load(xgb_pkl)

            if not hasattr(loaded_model, "gpu_id"):
                setattr(loaded_model, "gpu_id", None)
            if not hasattr(loaded_model, "predictor"):
                setattr(loaded_model, "predictor", None)

            self.sales_models["Xgboost"] = loaded_model

        for model_key in ["random_forest", "decision_tree"]:
            model_path = sales_dir / f"{model_key}.pkl"
            if model_path.exists():
                display_name = model_key.replace("_", " ").title()
                self.sales_models[display_name] = joblib.load(model_path)

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
        """Load feature scalers and target scalers for sales and quantity domains."""
        sales_scaler_path = self.models_dir / "sales_prediction" / "scalers" / "feature_scaler.pkl"
        qty_scaler_path = self.models_dir / "quantity_prediction" / "scalers" / "feature_scaler.pkl"

        if sales_scaler_path.exists():
            self.scalers["sales"] = joblib.load(sales_scaler_path)
            logger.info(f"✅ Loaded Sales Scaler from {sales_scaler_path}")

        if qty_scaler_path.exists():
            self.scalers["quantity"] = joblib.load(qty_scaler_path)
            logger.info(f"✅ Loaded Quantity Scaler from {qty_scaler_path}")

        sales_target_scaler_path = (
            self.models_dir / "sales_prediction" / "scalers" / "target_scaler.pkl"
        )
        qty_target_scaler_path = (
            self.models_dir / "quantity_prediction" / "scalers" / "target_scaler.pkl"
        )

        if sales_target_scaler_path.exists():
            self.target_scalers["sales"] = joblib.load(sales_target_scaler_path)
            logger.info(f"✅ Loaded Sales Target Scaler from {sales_target_scaler_path}")

        if qty_target_scaler_path.exists():
            self.target_scalers["quantity"] = joblib.load(qty_target_scaler_path)
            logger.info(f"✅ Loaded Quantity Target Scaler from {qty_target_scaler_path}")

    def _load_encoders(self) -> None:
        """Unpack nested categorical label encoders from single artifact dictionary."""
        encoder_path = self.models_dir / "sales_prediction" / "encoders" / "label_encoders.pkl"
        if not encoder_path.exists():
            logger.warning(f"⚠️ Label encoders artifact not found at {encoder_path}")
            return

        raw_encoders: Dict[str, Any] = joblib.load(encoder_path)
        for col_name, encoder_obj in raw_encoders.items():
            if isinstance(encoder_obj, dict):
                for val in encoder_obj.values():
                    if isinstance(val, LabelEncoder):
                        self.encoders[col_name] = val
                        break

            elif isinstance(encoder_obj, LabelEncoder):
                self.encoders[col_name] = encoder_obj

        logger.info(f"✅ Successfully loaded {len(self.encoders)} categorical label encoders.")

    # =========================================================================
    # 2. Feature Engineering & Preprocessing
    # =========================================================================

    def _find_reference_row(self, input_features: Dict[str, Any]) -> Optional[pd.Series]:
        """Find baseline matching row from reference dataset preserving numeric and categorical attributes."""
        if self.reference_df is None or self.reference_df.empty:
            return None

        df = self.reference_df.copy()
        matched = df

        if "car_id" in input_features and "car_id" in df.columns:
            filtered = df[
                df["car_id"].astype(str).str.lower() == str(input_features["car_id"]).lower()
            ]
            if not filtered.empty:
                matched = filtered
        elif "model" in input_features and "model" in df.columns:
            filtered = df[
                df["model"].astype(str).str.lower() == str(input_features["model"]).lower()
            ]
            if not filtered.empty:
                matched = filtered

        reference_dict: Dict[str, Any] = {}
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                reference_dict[col] = matched[col].median()
            else:
                mode_series = matched[col].mode()
                reference_dict[col] = (
                    mode_series.iloc[0] if not mode_series.empty else matched[col].iloc[0]
                )

        return pd.Series(reference_dict)

    def _build_feature_dataframe(
        self, input_features: Dict[str, Any], source_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """Construct complete single-row DataFrame using inputs, derived metrics, and reference values."""
        feature_dict = input_features.copy()

        now = pd.Timestamp.now()
        feature_dict.setdefault("day_of_week", now.dayofweek)
        feature_dict.setdefault("week_of_year", now.isocalendar().week)
        feature_dict.setdefault("season", (now.month % 12 + 3) // 3)

        # Auto-derive monetary metrics if missing
        if (
            "price" in feature_dict
            and "quantity" in feature_dict
            and "gross_sales" not in feature_dict
        ):
            feature_dict["gross_sales"] = float(feature_dict["price"]) * float(
                feature_dict["quantity"]
            )

        if (
            "profit" not in feature_dict
            and "gross_sales" in feature_dict
            and "cost" in feature_dict
        ):
            feature_dict["profit"] = float(feature_dict["gross_sales"]) - float(
                feature_dict["cost"]
            )

        feature_df = pd.DataFrame([feature_dict])

        reference_row = (
            source_df.iloc[0]
            if (source_df is not None and not source_df.empty)
            else self._find_reference_row(feature_dict)
        )

        if reference_row is not None:
            for col in reference_row.index:
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
                logger.warning(f"⚠️ Encoder for column '{col}' missing 'classes_'. Skipping.")
                continue

            series_str = df[col].astype(str)

            try:
                df[col] = encoder.transform(series_str)
            except ValueError:
                known_classes = set(encoder.classes_.tolist())
                fallback_class = encoder.classes_[0]
                cleaned_series = series_str.apply(
                    lambda x: x if x in known_classes else fallback_class
                )
                df[col] = encoder.transform(cleaned_series)
                logger.warning(f"⚠️ Unseen value in '{col}'. Fallback to '{fallback_class}'.")

        return df

    def _scale_and_align_features(
        self, df: pd.DataFrame, domain_type: str, selected_model: Any
    ) -> pd.DataFrame:
        """Align columns to match training schema and apply feature scaling."""
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass

        skewed_features = [
            "sales",
            "lag_1",
            "lag_7",
            "lag_30",
            "rolling_mean_7",
            "rolling_std_7",
            "rolling_max_7",
            "cost",
            "total_cost",
            "profit",
            "price",
            "income_customer",
            "gross_sales",
            "discount_amount",
        ]
        for col in df.columns:
            if col in skewed_features and pd.api.types.is_numeric_dtype(df[col]):
                df[col] = np.log1p(np.maximum(0.0, df[col]))

        expected_columns: List[str] = []

        scaler = self.scalers.get(domain_type)
        if isinstance(selected_model, xgb.Booster):
            expected_columns = selected_model.feature_names or []
        elif hasattr(selected_model, "feature_names_in_"):
            expected_columns = list(selected_model.feature_names_in_)
        elif scaler and hasattr(scaler, "feature_names_in_"):
            expected_columns = list(scaler.feature_names_in_)

        if expected_columns:
            missing = [c for c in expected_columns if c not in df.columns]
            if missing:
                logger.warning(
                    f"⚠️ Missing expected features for domain '{domain_type}': {missing}. Zero-filling."
                )
                for m_col in missing:
                    df[m_col] = 0

            df = df[expected_columns]
            logger.info(f"📊 Final Dataframe shape before scaling: {df.shape}")

        model_class_name = selected_model.__class__.__name__.lower()
        if scaler is not None:
            logger.info(f"📊 Applying feature scaler for model: {model_class_name}")
            scaled_array = scaler.transform(df)
            return pd.DataFrame(scaled_array, columns=df.columns)

        return df

    # =========================================================================
    # 3. Main Inference Pipeline
    # =========================================================================

    def _calculate_confidence_score(
        self,
        selected_model: Any,
        features_df: pd.DataFrame,
        raw_prediction: float,
        final_prediction: float = 0.0,
    ) -> float:
        """Calculate realistic confidence score using ensemble variance or target z-distance."""
        try:
            numeric_df = features_df.copy()
            for col in numeric_df.columns:
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
        except Exception as e:
            logger.warning(f"⚠️ Failed converting features for confidence score: {e}")
            return 0.50

        if hasattr(selected_model, "estimators_"):
            try:
                tree_predictions = np.array(
                    [tree.predict(features_df.values)[0] for tree in selected_model.estimators_]
                )
                std_dev = np.std(tree_predictions)
                mean_pred = np.mean(tree_predictions)

                if mean_pred == 0:
                    return 0.50

                cv = abs(std_dev / mean_pred)
                confidence = float(np.exp(-cv))
                return round(float(np.clip(confidence, 0.10, 0.99)), 3)

            except Exception as e:
                logger.warning(f"⚠️ Tree ensemble confidence error: {e}")

        if final_prediction < 0:
            return 0.15

        if self.reference_df is not None and not self.reference_df.empty:
            mean_val = (
                float(self.reference_df["sales"].mean())
                if "sales" in self.reference_df.columns
                else 10000.0
            )
            std_val = (
                float(self.reference_df["sales"].std())
                if "sales" in self.reference_df.columns
                else 5000.0
            )

            z_score = abs(final_prediction - mean_val) / (std_val + 1e-5)
            confidence = float(np.exp(-0.5 * z_score))
            return round(float(np.clip(confidence, 0.35, 0.95)), 3)

        return 0.75

    def _scale_prediction_to_range(self, raw_pred: float, domain_key: str) -> float:
        """Inverse-transform predictions based on target_scaler (StandardScaler) and expm1 log transform."""
        domain_type = "sales" if "sale" in domain_key.lower() else "quantity"
        target_scaler = self.target_scalers.get(domain_type)

        pred_log = raw_pred
        if target_scaler is not None:
            try:
                pred_log = float(target_scaler.inverse_transform(np.array([[raw_pred]]))[0][0])
                logger.info(
                    f"🔄 Applied target scaler inverse transform for {domain_type}: {raw_pred:.4f} -> {pred_log:.4f}"
                )
            except Exception as e:
                logger.warning(f"⚠️ Error applying target scaler inverse transform: {e}")

        # Revert log1p transformation: expm1(pred_log) = exp(pred_log) - 1
        if pred_log > 0.0:
            denormalized = float(np.expm1(pred_log))
            logger.info(
                f"🔄 Applied np.expm1 log transform for {domain_type}: {pred_log:.4f} -> {denormalized:.2f}"
            )
            return max(0.0, denormalized)

        return max(0.0, float(pred_log))

    def _predict_single_domain(
        self,
        domain_key: str,
        model_name: str,
        feature_dict: Dict[str, Any],
        source_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[float, float]:
        is_sales = "sale" in domain_key.lower()
        available_models = self.sales_models if is_sales else self.qty_models
        selected_model = available_models.get(model_name)

        if not selected_model:
            raise ValueError(
                f"Model '{model_name}' not found for domain '{domain_key}'. "
                f"Available models: {list(available_models.keys())}"
            )

        features_df = self._build_feature_dataframe(feature_dict, source_df)
        features_df_encoded = self._encode_categorical_features(features_df.copy())

        domain_type = "sales" if is_sales else "quantity"
        features_df_scaled = self._scale_and_align_features(
            features_df_encoded.copy(), domain_type, selected_model
        )

        if not hasattr(selected_model, "gpu_id"):
            setattr(selected_model, "gpu_id", None)
        if not hasattr(selected_model, "predictor"):
            setattr(selected_model, "predictor", None)

        # Model Inference
        if isinstance(selected_model, xgb.Booster):
            dmatrix = xgb.DMatrix(features_df_scaled)
            raw_pred = float(selected_model.predict(dmatrix)[0])
        else:
            raw_pred = float(selected_model.predict(features_df_scaled)[0])

        logger.info(f"📊 Raw [{domain_type}] prediction for '{model_name}': {raw_pred}")

        final_pred = self._scale_prediction_to_range(raw_pred, domain_type)

        confidence = self._calculate_confidence_score(
            selected_model, features_df_scaled, raw_pred, final_pred
        )

        return final_pred, confidence

    def predict(
        self, request: PredictRequest, source_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        raw_type = request.model_type.lower()
        sales_m_name = request.sales_model_name or request.model_name
        qty_m_name = request.quantity_model_name or request.model_name

        response_data: Dict[str, Any] = {
            "model_type": raw_type,
            "predicted_sales": None,
            "sales_confidence": None,
            "predicted_quantity": None,
            "quantity_confidence": None,
        }

        features = request.features.copy()
        initial_gross_sales = request.features.get("gross_sales")
        initial_profit = request.features.get("profit")

        # 1. Predict Quantity first
        if raw_type in ["quantity", "both", "all"]:
            raw_qty, qty_conf = self._predict_single_domain(
                "quantity", qty_m_name, features, source_df
            )
            response_data["predicted_quantity"] = max(1, int(round(raw_qty)))
            response_data["quantity_confidence"] = qty_conf

            # Pass updated quantity feature
            features["quantity"] = response_data["predicted_quantity"]

            # Only recalculate gross_sales if gross_sales was NOT explicitly provided by user
            if initial_gross_sales is None:
                if "price" in features:
                    features["gross_sales"] = float(features["price"]) * float(features["quantity"])
                    if "cost" in features:
                        features["profit"] = float(features["gross_sales"]) - float(
                            features["cost"]
                        )
            else:
                features["gross_sales"] = float(initial_gross_sales)
                if initial_profit is not None:
                    features["profit"] = float(initial_profit)
                elif "cost" in features:
                    features["profit"] = float(features["gross_sales"]) - float(features["cost"])

        # 2. Predict Sales with updated features
        if raw_type in ["sales", "both", "all"]:
            raw_sales, sales_conf = self._predict_single_domain(
                "sales", sales_m_name, features, source_df
            )
            response_data["predicted_sales"] = round(max(0.0, raw_sales), 2)
            response_data["sales_confidence"] = sales_conf

        return response_data
