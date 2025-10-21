import logging
import pandas as pd
from enum import Enum
from typing import List,Optional
from abc import ABC,abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from spark_session import get_or_create_spark_session
import joblib
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureScalingStrategy(ABC):

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
        self.fitted_model = None

    @abstractmethod
    def scale(self, df: pd.DataFrame, columns_to_scale: List[str]) -> pd.DataFrame:
        pass

class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self,spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.scaler_models = {}
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        ##################### Pandas code ##############
        
        # df[columns_to_scale]=self.scaler.fit_transform(df[columns_to_scale])
        # self.fitted = True
        # logging.info(f"Applied MinMax scaling to column: {columns_to_scale}")
        # path = "artifacts/encode/minmax_scaler.joblib"
        # joblib.dump(self.scaler, path)
        # return df
    
        #################### Pyspark code ###############
        df_scaled = df

        for col in columns_to_scale:
            vector_col = f"{col}_vec"
            assembler = VectorAssembler(inputCol=[col], outputCol=vector_col)

            scaled_vector_col = f"{col}_scaled_vec"
            scaler = MinMaxScaler(inputCol=vector_col, cutputCol=scaled_vector_col)

            pipeline = Pipeline(stages=[assembler, scaler])
            pipeline_model = pipeline.fit(df_scaled)

            get_value_udf = F.udf(lambda x: float(x[0] if x is not None else None), "double")
            df_scaled = df_scaled.withColumn(
                                            col,
                                            get_value_udf(F.col(scaled_vector_col))
                                            )
        return df_scaled

class StandardScalingStrategy(FeatureScalingStrategy):
    def __init__(self,spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.scaler_models = {}
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        ##################### Pandas code ##############
        
        # df[columns_to_scale]=self.scaler.fit_transform(df[columns_to_scale])
        # self.fitted = True
        # logging.info(f"Applied MinMax scaling to column: {columns_to_scale}")
        # path = "artifacts/encode/minmax_scaler.joblib"
        # joblib.dump(self.scaler, path)
        # return df
    
        #################### Pyspark code ###############
        df_scaled = df

        for col in columns_to_scale:
            vector_col = f"{col}_vec"
            assembler = VectorAssembler(inputCol=[col], outputCol=vector_col)

            scaled_vector_col = f"{col}_scaled_vec"
            scaler = StandardScaler(inputCol=vector_col, cutputCol=scaled_vector_col)

            pipeline = Pipeline(stages=[assembler, scaler])
            pipeline_model = pipeline.fit(df_scaled)

            get_value_udf = F.udf(lambda x: float(x[0] if x is not None else None), "double")
            df_scaled = df_scaled.withColumn(
                                            col,
                                            get_value_udf(F.col(scaled_vector_col))
                                            )
        return df_scaled

    def get_scaler(self):
        return self.scaler