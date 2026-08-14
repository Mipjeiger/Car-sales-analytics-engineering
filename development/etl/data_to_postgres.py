import pandas as pd
import logging
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pathlib import Path

ENV_DIR = Path(__file__).resolve().parent.parent.parent / "development" / ".env"
load_dotenv(dotenv_path=ENV_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration
PARQUET_PATH = (
    Path(__file__).resolve().parent.parent.parent / "development" / "database" / "car_sales.parquet"
)
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_HOST = os.getenv("POSTGRES_HOST")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")
POSTGRES_DB = os.getenv("POSTGRES_DB")
SCHEMA_NAME = os.getenv("POSTGRES_SCHEMA_NAME")
TABLE_NAME = os.getenv("POSTGRES_TABLE_NAME")

# Build function


def create_db_engine():
    """Create a SQLALchemy engine for PostgreSQL."""
    try:
        engine = f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
        engine = create_engine(engine)
        return engine

    except SQLAlchemyError as e:
        logger.error(f"Error creating database engine: {e}")
        raise RuntimeError(f"Error creating database engine: {e}")


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and prepare data for PostgreSQL"""
    try:
        # Check the problematic columns
        logger.info("Checking for remaining 'None' values...")
        for col in df.columns:
            if df[col].dtype == "object":
                none_count = (df[col] == "None").sum()
                if none_count > 0:
                    logger.warning(f"Column '{col}' has {none_count} 'None' values remaining.")

            if df[col].isnull().sum() > 0:
                logger.info(f"Column '{col}' has {df[col].isnull().sum()} null values.")

        # 1. Replace all common null-like strings
        null_values = ["None", "none", "NULL", "null", "Null", "NaN", "nan", "N/A", "n/a", ""]
        df = df.replace(null_values, np.nan)

        # 2. Convert date
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # 3. Integer columns → use nullable Int64
        int_columns = ["day", "month", "year", "phone", "quantity"]
        for col in int_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # 4. Float columns
        float_columns = [
            "price",
            "income_customer",
            "discount",
            "gross_sales",
            "discount_amount",
            "sales",
            "cost",
            "total_cost",
            "profit",
            "profit_margin",
        ]
        for col in float_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 5. Force object columns to string (optional but safer)
        object_columns = df.select_dtypes(include=["object"]).columns
        for col in object_columns:
            df[col] = df[col].astype(str).replace("nan", np.nan)

        return df

    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        raise


def load_db_to_postgres():
    """Load data from Parquet file to PostgreSQL."""
    try:
        # 1. Read parquet file into a DataFrame
        logger.info(f"Reading Parquet file from {PARQUET_PATH}")
        df = pd.read_parquet(PARQUET_PATH)
        logger.info(f"Successfully read Parquet file with {len(df)} records.")

        # Load clean feature engineered data
        logger.info("Performing feature engineering on the DataFrame.")
        df = feature_engineering(df)
        logger.info("Feature engineering completed successfully.")

        # 2. Create database engine
        engine = create_db_engine()

        # 3. Load DataFrame to PostgreSQL
        logger.info(f"Loading data into PostgreSQL table: {TABLE_NAME}")
        df.to_sql(
            name=TABLE_NAME,
            con=engine,
            schema=SCHEMA_NAME,
            if_exists="replace",
            index=False,
            method=None,
            chunksize=1000,  # Adjust chunk size as needed
        )
        logger.info(f"✅ Successfully loaded data into PostgreSQL table: {TABLE_NAME}")

        # 4. Verify the number of records in the PostgreSQL table
        with engine.connect() as conn:
            result = conn.execute(text(f'SELECT COUNT(*) FROM "{SCHEMA_NAME}"."{TABLE_NAME}"'))
            count = result.scalar()
            logger.info(f"✅ Successfully loaded {count:,} rows into '{TABLE_NAME}'")

    except FileNotFoundError:
        logger.error(f"⚠️ Parquet file not found at {PARQUET_PATH}. Please check the file path.")
    except SQLAlchemyError as e:
        logger.error(f"❌ Database error: {e}")
    except Exception as e:
        logger.error(f" ❌ An unexpected error occurred: {e}")


if __name__ == "__main__":
    load_db_to_postgres()
