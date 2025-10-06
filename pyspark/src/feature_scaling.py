import logging
import pandas as pd
import numpy as np
from enum import Enum
from typing import List,Optional
from abc import ABC, abstractmethod
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import MinMaxScaler, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from spark_session import get_or_create_spark_session
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


class ScalingType(str, Enum):
    MINMAX = 'minmax'
    STANDARD = 'standard'

class MinMaxScalingStrategy(FeatureScalingStrategy):
    def __init__(self,output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.output_col_suffix = output_col_suffix
        self.scaler_models = {}
    
    def scale(self,df,columns_to_scale: List[str]):
        for col in columns_to_scale:
            vector_col = f"{col}_vec"
            assembler = VectorAssembler(inputCols=[col], cutputCol=vector_col)

            scaled_vector_col = f"{col}_scaled_vec"
            scaler = MinMaxScaler(inputCols=vector_col, cutputCol=scaled_vector_col)

            pipeline = Pipeline(stages=[assembler, scaler])
            pipeline_model = pipeline.fit(df_scaled)

            get_value_udf = F.udf(lambda x: float(x[0] if x is not None else None), "double")
            df_scaled = df_scaled.withColumn(
                                            col,
                                            get_value_udf(F.col(scaled_vector_col))
                                            )

        return df_scaled
    
class StandardScalingStrategy(FeatureScalingStrategy):
    """Standard scaling strategy to scale features to zero mean and unit variance."""
    
    def __init__(self, with_mean: bool = True, with_std: bool = True, 
                 output_col_suffix: str = "_scaled", spark: Optional[SparkSession] = None):

        super().__init__(spark)
        self.with_mean = with_mean
        self.with_std = with_std
        self.output_col_suffix = output_col_suffix
        self.scaler_models = {}
    
    def scale(self, df: DataFrame, columns_to_scale: List[str]) -> DataFrame:
        df_scaled = df 

        for col in columns_to_scale:
            vector_col = f"{col}_vec"
            assembler = VectorAssembler(inputCols=[col], cutputCol=vector_col)

            scaled_vector_col = f"{col}_scaled_vec"
            scaler = StandardScaler(inputCols=vector_col, cutputCol=scaled_vector_col)

            pipeline = Pipeline(stages=[assembler, scaler])
            pipeline_model = pipeline.fit(df_scaled)

            get_value_udf = F.udf(lambda x: float(x[0] if x is not None else None), "double")
            df_scaled = df_scaled.withColumn(
                                            col,
                                            get_value_udf(F.col(scaled_vector_col))
                                            )

        return df_scaled