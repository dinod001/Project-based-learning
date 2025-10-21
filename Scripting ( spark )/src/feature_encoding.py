import logging
import os
import pandas as pd
import json
from enum import Enum
from abc import ABC,abstractmethod
from typing import Dict,List,Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, OneHotEncoder, IndexToString
from pyspark.ml import Pipeline
from spark_session import get_or_create_spark_session
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEncodingStrategy(ABC):
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self,nominal_columns,spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.nominal_columns = nominal_columns
        self.encoder_dicts = {}
        self.indexers = {}
        self.encoders = {}
        os.makedirs('artifacts/encode', exist_ok=True)
    
    def encode(self, df):
        """
        Apply simple one-hot encoding to all nominal columns.
        """
        df_encoded = df

        for col in self.nominal_columns:
            # Step 1: Convert text → index
            indexer = StringIndexer(inputCol=col, outputCol=f"{col}_index", handleInvalid="keep")
            model = indexer.fit(df_encoded)
            labels = model.labels
            df_encoded = model.transform(df_encoded)

            # Step 2: Convert index → one-hot vector
            encoder = OneHotEncoder(inputCol=f"{col}_index", outputCol=f"{col}_vec", dropLast=False)
            df_encoded = encoder.fit(df_encoded).transform(df_encoded)

            # Step 3: Convert vector → array, then split into multiple binary columns
            df_encoded = df_encoded.withColumn("tmp_array", F.expr(f"{col}_vec.toArray()"))

            for i, label in enumerate(labels):
                clean_label = label.replace(" ", "_")
                df_encoded = df_encoded.withColumn(f"{col}_{clean_label}", F.col("tmp_array")[i].cast("int"))

            # Step 4: Drop original and temp columns
            df_encoded = df_encoded.drop(col, f"{col}_index", f"{col}_vec", "tmp_array")

        return df_encoded

class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings: Dict[str, Dict[str, int]], spark: Optional[SparkSession] = None):
        super.__init__(spark)
        self.ordinal_mappings = ordinal_mappings
    
    def encode(self,df):
        #################### Pandas Code #####################
        # for column,mapping in self.ordinal_mappings.items():
        #     df[column] = df[column].map(mapping)
        
        # return df

        ################## Pyspark Code #######################
        df_encoded = df

        for column, mapping in self.ordinal_mappings.items():
                mapping_expr = F.when(F.col(column).isNull(), None)
                for value, code in mapping.item():
                    mapping_expr = mapping_expr.when(F.col(column) == value, code)

                df_encoded = df_encoded.withColumn(column, mapping_expr)

        return df_encoded
    
