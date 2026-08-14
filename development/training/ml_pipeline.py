import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import warnings
import logging
from utils import load_data, prepare_features, save_model_to_minio, save_metrics_to_minio

warnings.filterwarnings("ignore")

# Logging configuration
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MLTrainer:
    """
    ML Training Pipeline with RandomizedSearchCV
    Trains: CatBoost, Decision Tree, Random Forest, XGBoost
    Targets: Sales, Quantity
    Saves models to MinIO
    """

    def __init__(self):
        self.df = load_data()
        self.results = []

    def train_sales_models(self):
        """Train models for sales prediction"""
        X, y = prepare_features(self.df, target="sales")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "XGBoost": {
                "model": XGBRegressor(random_state=42, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.6, 0.8, 1.0],
                },
            },
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "Decision Tree": {
                "model": DecisionTreeRegressor(random_state=42),
                "params": {
                    "max_depth": [5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "CatBoost": {
                "model": CatBoostRegressor(random_state=42),
                "params": {
                    "iterations": [100, 200, 300],
                    "depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "l2_leaf_reg": [1, 3, 5],
                },
            },
        }

        for name, config in models.items():
            logger.info(f"Training {name} for sales prediction...")
            search = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=10,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            search.fit(X_train, y_train)

            # Predict on test set
            y_pred = search.best_estimator_.predict(X_test)
            metrics = {
                "Model": name,
                "Type": "Sales",
                "R2": r2_score(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred),
            }
            # Store metrics into results
            self.results.append(metrics)

            # Save model to MinIO
            save_model_to_minio(search.best_estimator_, name, "sales")
            logger.info(f"📂 {name} model for sales prediction saved to MinIO.")
            logger.info(f"✅ {name} Sales: R2={metrics['R2']:.4f}")

        # Save metrics to MinIO
        df_results = pd.DataFrame([r for r in self.results if r["Type"] == "Sales"])
        save_metrics_to_minio(df_results, "sales")

    def train_quantity_models(self):
        """Train models for quantity prediction"""
        X, y = prepare_features(self.df, target="quantity")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "XGBoost": {
                "model": XGBRegressor(random_state=42, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.6, 0.8, 1.0],
                },
            },
            "Random Forest": {
                "model": RandomForestRegressor(random_state=42, n_jobs=-1),
                "params": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "Decision Tree": {
                "model": DecisionTreeRegressor(random_state=42),
                "params": {
                    "max_depth": [5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
            },
            "CatBoost": {
                "model": CatBoostRegressor(random_state=42),
                "params": {
                    "iterations": [100, 200, 300],
                    "depth": [3, 5, 7, 9],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "l2_leaf_reg": [1, 3, 5],
                },
            },
        }

        for name, config in models.items():
            logger.info(f"Training {name} for quantity prediction...")
            search = RandomizedSearchCV(
                config["model"],
                config["params"],
                n_iter=10,
                cv=5,
                scoring="r2",
                n_jobs=-1,
                random_state=42,
                verbose=1,
            )
            search.fit(X_train, y_train)

            # Predict on test set
            y_pred = search.best_estimator_.predict(X_test)
            metrics = {
                "Model": name,
                "Type": "Quantity",
                "R2": r2_score(y_test, y_pred),
                "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                "MAE": mean_absolute_error(y_test, y_pred),
                "MAPE": mean_absolute_percentage_error(y_test, y_pred),
            }
            # Store metrics into results
            self.results.append(metrics)

            # Save model to MinIO
            save_model_to_minio(search.best_estimator_, name, "quantity")
            logger.info(f"📂 {name} model for quantity prediction saved to MinIO.")
            logger.info(f"✅ {name} Quantity: R2={metrics['R2']:.4f}")

        # Save metrics to MinIO
        df_results = pd.DataFrame([r for r in self.results if r["Type"] == "Quantity"])
        save_metrics_to_minio(df_results, "quantity")

    def run(self):
        """Run all training pipelines"""
        logger.info("🚀 Starting ML Training Pipeline...")
        self.train_sales_models()
        self.train_quantity_models()

        # Save all results
        df_results = pd.DataFrame(self.results)
        save_metrics_to_minio(df_results, "all")

        logger.info("✅ ML Training Pipeline completed successfully.")
        print(df_results.to_string())


# Run the training pipeline
if __name__ == "__main__":
    trainer = MLTrainer()
    trainer.run()
