import os
import sys
import logging
import pandas as pd
from typing import Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from spark_session import create_spark_session, stop_spark_session
from spark_utils import save_dataframe, spark_to_pandas, get_dataframe_info, check_missing_values
from data_ingestion import DataIngestorCSV
from handling_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from data_splitter import SimpleTrainTestSplitStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config
from mlflow_utils import MLflowTracker, setup_mlflow_autolog, create_mlflow_run_tags
import mlflow

def data_pipeline(
    data_path: str = 'data/raw/ChurnModelling.csv',
    )->Dict[str, np.ndarray]:

    # Initialize Spark session
    spark = create_spark_session("ChurnPredictionDataPipeline")

    data_paths=get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scalling_config = get_scaling_config()
    splitting_config = get_splitting_config()

    #mlflow intergrating
    mlflow_tracker = MLflowTracker()
    setup_mlflow_autolog()
    run_tags = create_mlflow_run_tags(
                                'data_pipeline',{
                                    'data_paths':data_path
                                }
                                )
    
    mlflow_tracker.start_run(run_name='data_pipeline',tags=run_tags)

    print("step 01 data ingestion")

    artifacts_dir = os.path.join(os.path.dirname(__file__),'..',data_paths['data_artifacts_dir'])

    X_train_path = os.path.join(artifacts_dir,'X_train.csv')
    X_test_path = os.path.join(artifacts_dir,'X_test.csv')
    Y_train_path = os.path.join(artifacts_dir,'Y_train.csv')
    Y_test_path = os.path.join(artifacts_dir,'Y_test.csv')

    if os.path.exists(X_train_path) and \
       os.path.exists(X_test_path) and \
       os.path.exists(Y_train_path) and \
       os.path.exists(Y_test_path):

       X_train = pd.read_csv(X_train_path)
       X_test = pd.read_csv(X_test_path)
       Y_train = pd.read_csv(Y_train_path)
       Y_test = pd.read_csv(Y_test_path)
    
       mlflow_tracker.log_data_pipeline_metrics({
                        'total_rows': len(X_train) + len(X_test),
                        'train_rows': len(X_train),
                        'test_rows': len(X_test),
                        'num_features': X_train.shape[1],
                        'missing_values': X_train.isna().sum().sum(),
                        'outliers_removed': 0  # or however many you removed
                    })

    mlflow_tracker.end_run()

    os.makedirs(data_paths['data_artifacts_dir'],exist_ok=True)

    if not os.path.exists('temp_imputed.csv'):

        ingestor = DataIngestorCSV(spark)
        df = ingestor.ingest(data_path)

        print("\n step 02: Handle missing values")

        drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'],spark=spark)

        age_handler = FillMissingValuesStrategy(
                                            method='mean',
                                            relavant_column='Age',
                                            spark=spark
                                        )
        
        gender_handler = FillMissingValuesStrategy(
                                        relavant_column='Gender',
                                        is_custom_computer = True,
                                        custom_imputer = GenderImputer(),
                                        spark=spark
                                    )
        
        df = drop_handler.handle(df)
        df = age_handler.handle(df)
        df = gender_handler.handle(df)

        df.to_csv('temp_imputed.csv', index=False)
    
    df = pd.read_csv('temp_imputed.csv')

    print(f"data set shape after imputation {df.shape}")
    print(df.isnull().sum())

    print("\nStep 03: handling outliers")

    # Convert pandas DataFrame back to PySpark DataFrame for outlier detection
    df_spark = spark.createDataFrame(df)

    outlier_detector = OutlierDetector(strategy=IQROutlierDetection(spark=spark))

    df_spark = outlier_detector.handle_outliers(df_spark,columns['outlier_columns'])
    
    # Convert back to pandas DataFrame for subsequent steps
    df = df_spark.toPandas()

    # Drop temporary outlier detection columns
    outlier_cols_to_drop = [f"{col}_outlier" for col in columns['outlier_columns']]
    df = df.drop(columns=outlier_cols_to_drop)

    print(f"Data set shape after outlier removal {df.shape}")

    print("\nStep 04: Fetaure Binning")

    # Convert pandas DataFrame to PySpark DataFrame for binning
    df_spark = spark.createDataFrame(df)
    binning = CustomBinningStrategy(binning_config['credit_score_bins'],spark=spark)
    df_spark = binning.bin_feature(df_spark,'CreditScore')
    df = df_spark.toPandas()

    print(f"Data set shape after feature binning {df.shape}")

    print("\nStep 05: Fetaure Encoding")

    # Convert pandas DataFrame to PySpark DataFrame for encoding
    df_spark = spark.createDataFrame(df)
    nominal_startegy = NominalEncodingStrategy(encoding_config['nominal_columns'],spark=spark)
    ordinal_startegy = OrdinalEncodingStrategy(encoding_config['ordinal_mappings'],spark=spark)

    df_spark = nominal_startegy.encode(df_spark)
    df_spark = ordinal_startegy.encode(df_spark)
    df = df_spark.toPandas()

    print(f"Data set shape after encoding {df.shape}")

    print(df.head())

    print("\nStep 06: Fetaure Scalling")

    # Convert pandas DataFrame to PySpark DataFrame for scaling
    df_spark = spark.createDataFrame(df)
    minmax_strategy = MinMaxScalingStrategy(spark=spark)
    df_spark = minmax_strategy.scale(df_spark,scalling_config['columns_to_scale'])
    df = df_spark.toPandas()

    print(f"Data set shape after scalling {df.shape}")
    print(df.head())

    df = df.drop(columns=['RowNumber','CustomerId','Firstname','Lastname'])
    print(df)

    print("\nStep 07: Data Splitting")

    # Use pandas for data splitting to avoid PySpark Python worker issues
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    X = df.drop(columns=['Exited'])
    y = df['Exited']
    
    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, 
        test_size=splitting_config['test_size'], 
        random_state=splitting_config['random_state'],
        stratify=y  # Ensure balanced splits
    )
    
    # Convert to DataFrames for consistency
    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    Y_train = pd.DataFrame(Y_train, columns=['Exited'])
    Y_test = pd.DataFrame(Y_test, columns=['Exited'])

    X_train.to_csv(X_train_path,index=False)
    X_test.to_csv(X_test_path,index=False)
    Y_train.to_csv(Y_train_path,index=False)
    Y_test.to_csv(Y_test_path,index=False)

    print(f"X train shape {X_train.shape}")
    print(f"X test shape {X_test.shape}")
    print(f"Y train shape {Y_train.shape}")
    print(f"Y test shape {Y_test.shape}")

    # Stop Spark session
    stop_spark_session(spark)

data_pipeline()
