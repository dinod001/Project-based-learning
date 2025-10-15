import os
import sys
import logging
import pandas as pd
import numpy as np
import json
from typing import Dict

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Import project modules ===
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handling_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from data_splitter import SimpleTrainTestSplitStrategy
from spark_session import create_spark_session, stop_spark_session
from spark_utils import spark_to_pandas, save_dataframe

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import (
    get_data_paths, get_columns, get_outlier_config, get_binning_config,
    get_encoding_config, get_scaling_config, get_splitting_config
)
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
import mlflow


def data_pipeline(
        data_path: str = 'data/raw/ChurnModelling.csv',
        target_column: str = 'Exited',
        test_size: float = 0.2,
        force_rebuild: bool = False,
        output_format: str = "both"   # support both CSV and Parquet
    ) -> Dict[str, np.ndarray]:
    """
    Simplified PySpark + MLflow data preprocessing pipeline.
    """

    # === Load configs ===
    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()

    # === Initialize MLflow tracking ===
    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()
    run_tags = create_mlflow_run_tags('data_pipeline_pyspark', {'data_source': data_path})
    run = mlflow_tracker.start_run(run_name='data_pipeline_pyspark', tags=run_tags)

    # === Prepare output paths ===
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', data_paths['data_artifacts_dir'])
    os.makedirs(artifacts_dir, exist_ok=True)

    X_train_path = os.path.join(artifacts_dir, 'X_train.csv')
    X_test_path = os.path.join(artifacts_dir, 'X_test.csv')
    Y_train_path = os.path.join(artifacts_dir, 'Y_train.csv')
    Y_test_path = os.path.join(artifacts_dir, 'Y_test.csv')

    # === Reuse existing processed artifacts if available ===
    if all(os.path.exists(p) for p in [X_train_path, X_test_path, Y_train_path, Y_test_path]) and not force_rebuild:
        logger.info("✅ Using existing processed data artifacts")

        X_train = pd.read_csv(X_train_path)
        X_test = pd.read_csv(X_test_path)
        Y_train = pd.read_csv(Y_train_path)
        Y_test = pd.read_csv(Y_test_path)

        mlflow_tracker.log_data_pipeline_metrics({
            'total_samples': len(X_train) + len(X_test),
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'engine': 'existing_artifacts'
        })
        mlflow_tracker.end_run()
        return {
            'X_train': X_train.values,
            'X_test': X_test.values,
            'Y_train': Y_train.values.ravel(),
            'Y_test': Y_test.values.ravel()
        }

    # === Initialize Spark session ===
    spark = create_spark_session("ChurnDataPipeline")
    logger.info("🚀 Spark session created")

    try:
        # === Step 1: Data Ingestion ===
        logger.info("Step 1: Data Ingestion")
        ingestor = DataIngestorCSV(spark)
        df = ingestor.ingest(data_path)
        logger.info(f"Loaded data: {df.count()} rows, {len(df.columns)} columns")

        # === Step 2: Handle Missing Values ===
        logger.info("Step 2: Handling Missing Values")
        drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'], spark=spark)
        age_handler = FillMissingValuesStrategy(method='mean', relevant_column='Age', spark=spark)
        df = drop_handler.handle(df)
        df = age_handler.handle(df)
        df = df.fillna({'Gender': 'Unknown'})
        logger.info("Missing values handled")

        # === Step 3: Handle Outliers ===
        logger.info("Step 3: Outlier Detection")
        outlier_detector = OutlierDetector(strategy=IQROutlierDetection(spark=spark))
        df = outlier_detector.handle_outliers(df, columns['outlier_columns'])
        logger.info("Outliers removed")

        # === Step 4: Feature Binning ===
        logger.info("Step 4: Feature Binning")
        binning = CustomBinningStrategy(binning_config['credit_score_bins'], spark=spark)
        df = binning.bin_feature(df, 'CreditScore')
        logger.info("Feature binning complete")

        # === Step 5: Feature Encoding ===
        logger.info("Step 5: Feature Encoding")
        nominal_encoder = NominalEncodingStrategy(encoding_config['nominal_columns'], spark=spark)
        ordinal_encoder = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'], spark=spark)
        df = nominal_encoder.encode(df)
        df = ordinal_encoder.encode(df)
        logger.info("Encoding complete")

        # === Step 6: Feature Scaling ===
        logger.info("Step 6: Feature Scaling")
        minmax_scaler = MinMaxScalingStrategy(spark=spark)
        df = minmax_scaler.scale(df, scaling_config['columns_to_scale'])
        logger.info("Scaling complete")

        # === Step 7: Drop unnecessary columns ===
        drop_cols = ['RowNumber', 'CustomerId', 'Firstname', 'Lastname']
        existing = [c for c in drop_cols if c in df.columns]
        if existing:
            df = df.drop(*existing)
            logger.info(f"Dropped columns: {existing}")

        # === Step 8: Train-Test Split ===
        logger.info("Step 7: Splitting Data")
        splitter = SimpleTrainTestSplitStrategy(test_size=splitting_config['test_size'], spark=spark)
        X_train, X_test, Y_train, Y_test = splitter.split_data(df, target_column)

        # === Step 9: Save Outputs ===
        if output_format in ["csv", "both"]:
            spark_to_pandas(X_train).to_csv(X_train_path, index=False)
            spark_to_pandas(X_test).to_csv(X_test_path, index=False)
            spark_to_pandas(Y_train).to_csv(Y_train_path, index=False)
            spark_to_pandas(Y_test).to_csv(Y_test_path, index=False)

        if output_format in ["parquet", "both"]:
            save_dataframe(X_train, os.path.join(artifacts_dir, 'X_train.parquet'), format='parquet')
            save_dataframe(X_test, os.path.join(artifacts_dir, 'X_test.parquet'), format='parquet')
            save_dataframe(Y_train, os.path.join(artifacts_dir, 'Y_train.parquet'), format='parquet')
            save_dataframe(Y_test, os.path.join(artifacts_dir, 'Y_test.parquet'), format='parquet')

        logger.info(f"✅ Data saved successfully to {artifacts_dir}")

        # === Step 10: Log metrics to MLflow ===
        mlflow_tracker.log_data_pipeline_metrics({
            'train_samples': X_train.count(),
            'test_samples': X_test.count(),
            'features': len(X_train.columns),
            'engine': 'pyspark'
        })
        mlflow.log_params({'pipeline_version': '3.0_pyspark_simple'})

        # === Convert to numpy for return ===
        X_train_np = spark_to_pandas(X_train).values
        X_test_np = spark_to_pandas(X_test).values
        Y_train_np = spark_to_pandas(Y_train).values.ravel()
        Y_test_np = spark_to_pandas(Y_test).values.ravel()

        mlflow_tracker.end_run()
        logger.info("🎯 PySpark pipeline completed successfully")

        return {
            'X_train': X_train_np,
            'X_test': X_test_np,
            'Y_train': Y_train_np,
            'Y_test': Y_test_np
        }

    except Exception as e:
        logger.error(f"❌ Pipeline failed: {str(e)}")
        mlflow_tracker.end_run()
        raise

    finally:
        stop_spark_session(spark)
        logger.info("🧹 Spark session stopped")
