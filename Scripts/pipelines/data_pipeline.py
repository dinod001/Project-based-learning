import os
import sys
import logging
import pandas as pd
from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from data_ingestion import DataIngestorCSV
from handling_missing_values import DropMissingValuesStrategy, FillMissingValuesStrategy, GenderImputer
from outlier_detection import OutlierDetector, IQROutlierDetection
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy, NominalEncodingStrategy
from feature_scaling import MinMaxScalingStrategy
from data_splitter import SimpleTrainTestSplitStratergy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_data_paths, get_columns, get_missing_values_config, get_outlier_config, get_binning_config, get_encoding_config, get_scaling_config, get_splitting_config


def data_pipeline(
        data_path: str = 'data/raw/ChurnModelling.csv',
        target_column: str = 'Exited',
        test_size: float = 0.2,
        force_rebuild: bool = False
    ) -> Dict[str, np.ndarray]:

    data_paths = get_data_paths()
    columns = get_columns()
    outlier_config = get_outlier_config()
    binning_config = get_binning_config()
    encoding_config = get_encoding_config()
    scaling_config = get_scaling_config()
    splitting_config = get_splitting_config()


    print("Step 01 data ingestion")

    artifacts_dir = os.path.join(os.path.dirname(__file__),'..',data_paths['data_artifacts_dir'])

    X_train_path= os.path.join(artifacts_dir,'X_train.csv')
    X_test_path= os.path.join(artifacts_dir,'X_test.csv')
    Y_train_path= os.path.join(artifacts_dir,'Y_train.csv')
    Y_test_path= os.path.join(artifacts_dir,'Y_test.csv')

    if os.path.exists(X_train_path) and \
       os.path.exists(X_test_path) and \
       os.path.exists(Y_train_path) and \
       os.path.exists(Y_test_path):

       X_train = pd.read_csv(X_train_path)
       X_test = pd.read_csv(X_test_path)
       Y_train = pd.read_csv(Y_train_path)
       Y_test = pd.read_csv(Y_test_path)

    if not os.path.exists('temp_imputed.csv'):
        ingestor = DataIngestorCSV()
        df = ingestor.ingest(data_path)

        print("\nStep 02: Handle missing values")

        drop_handler = DropMissingValuesStrategy(critical_columns=columns['critical_columns'])

        age_handler = FillMissingValuesStrategy(
                                                method='mean',
                                                relavant_column='Age'
                                            )
        
        gender_handler = FillMissingValuesStrategy(
                                                    relavant_column='Gender',
                                                    is_custom_computer = True,
                                                    custom_imputer = GenderImputer()
                                                )
        df = drop_handler.handle(df)
        df = age_handler.handle(df)
        df = gender_handler.handle(df)

        df.to_csv('temp_imputed.csv', index=False)

    df = pd.read_csv('temp_imputed.csv')

    print(f"Data set shape after imputation {df.shape}")
    print(df.isnull().sum())

    print("\nStep 03: handling outliers")

    outlier_detector = OutlierDetector(strategy=IQROutlierDetection())

    df = outlier_detector.handle_outliers(df,columns['outlier_columns'])

    print(f"Data set shape after outlier removal {df.shape}")

data_pipeline()