import os
import sys
import pandas as pd
import numpy as np
from data_pipeline import data_pipeline
from typing import Dict, Any, Optional
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from spark_session import create_spark_session, stop_spark_session
from spark_utils import spark_to_pandas

from model_training import ModelTrainer
from model_evaluation import ModelEvaluator
from model_building import XGboostModelBuilder, RandomForestModelBuilder
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from mlflow_utils import MLflowTracker, create_mlflow_run_tags
from config import get_model_config, get_data_paths
import mlflow


def training_pipeline(
                    data_path: str = 'data/raw/ChurnModelling.csv',
                    model_params: Optional[Dict[str, Any]] = None,
                    test_size: float = 0.2, random_state: int = 42,
                    model_path: str = 'artifacts/models/churn_analysis.joblib',
                    data_format: str = 'csv'
                    ):
    """
    Execute comprehensive model training pipeline.
    """

    data_pipeline()
    spark = create_spark_session("ChurnPredictionTrainingPipeline")
    
    try:
        mlflow_tracker = MLflowTracker()
        run_tags = create_mlflow_run_tags(
            'training_pipeline', {
                'model_type': 'XGboost',
                'training_strategy': 'simple',
                'data_path': data_path,
                'model_path': model_path,
                'data_format': data_format,
                'processing_engine': 'pyspark'
            }
        )

        run = mlflow_tracker.start_run(run_name='training_pipeline', tags=run_tags)
        run_artifacts_dir = os.path.join('artifacts', 'mlflow_training_artifacts', run.info.run_id)
        os.makedirs(run_artifacts_dir, exist_ok=True)

        data_paths = get_data_paths()

        if data_format == 'parquet':
            X_train_spark = spark.read.parquet(f"{data_paths['processed_data']}/X_train.parquet")
            Y_train_spark = spark.read.parquet(f"{data_paths['processed_data']}/Y_train.parquet")
            X_test_spark = spark.read.parquet(f"{data_paths['processed_data']}/X_test.parquet")
            Y_test_spark = spark.read.parquet(f"{data_paths['processed_data']}/Y_test.parquet")
        else:
            X_train_spark = spark.read.csv(data_paths['X_train'], header=True, inferSchema=True)
            Y_train_spark = spark.read.csv(data_paths['Y_train'], header=True, inferSchema=True)
            X_test_spark = spark.read.csv(data_paths['X_test'], header=True, inferSchema=True)
            Y_test_spark = spark.read.csv(data_paths['Y_test'], header=True, inferSchema=True)

        X_train = spark_to_pandas(X_train_spark)
        Y_train = spark_to_pandas(Y_train_spark)
        X_test = spark_to_pandas(X_test_spark)
        Y_test = spark_to_pandas(Y_test_spark)

        mlflow.log_metrics({
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_features': X_train.shape[1],
            'train_class_0': (Y_train == 0).sum().iloc[0],
            'train_class_1': (Y_train == 1).sum().iloc[0],
            'test_class_0': (Y_test == 0).sum().iloc[0],
            'test_class_1': (Y_test == 1).sum().iloc[0]
        })

        mlflow.log_param('feature_names', list(X_train.columns))

        import time
        training_start_time = time.time()

        model_builder = XGboostModelBuilder(**model_params)
        model = model_builder.build_model()

        trainer = ModelTrainer()
        model, training_history = trainer.train(
            model=model,
            X_train=X_train,
            Y_train=Y_train.squeeze()
        )

        training_end_time = time.time()
        training_time = training_end_time - training_start_time

        trainer.save_model(model, model_path)
        mlflow.log_artifact(model_path, "trained_models")

        evaluator = ModelEvaluator(model, 'XGboost')
        evaluation_results = evaluator.evaluate(X_test, Y_test)
        evaluation_results_cp = evaluation_results.copy()

        if 'cm' in evaluation_results_cp:
            del evaluation_results_cp['cm']

        evaluation_results_cp.update({
            'training_time_seconds': training_time,
            'model_complexity': model.n_estimators if hasattr(model, 'n_estimators') else 0,
            'max_depth': model.max_depth if hasattr(model, 'max_depth') else 0
        })

        model_config = get_model_config()['model_params']
        mlflow_tracker.log_training_metrics(model, evaluation_results_cp, model_config)

        training_summary = {
            'model_type': 'XGboost',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': X_train.shape[1],
            'training_time': training_time,
            'model_path': model_path,
            'performance_metrics': evaluation_results_cp,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        summary_path = os.path.join(run_artifacts_dir, 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2, default=str)

        mlflow.log_artifact(summary_path, "training_summary")
        mlflow_tracker.end_run()

    except Exception as e:
        if 'mlflow_tracker' in locals():
            mlflow_tracker.end_run()
        raise
    finally:
        stop_spark_session(spark)


if __name__ == '__main__':
    model_config = get_model_config()
    model_params = model_config.get('model_params')
    training_pipeline(model_params=model_params)
