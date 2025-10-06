import logging
import pandas as pd
import os
import json
from enum import Enum
from typing import Dict, List,Optional
from abc import ABC, abstractmethod
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


class VariableType(str, Enum):
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'

class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self,nominal_columns, one_hot: bool = False, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.nominal_columns = nominal_columns
        self.one_hot = one_hot
        self.encoder_dicts = {}
        self.indexers = {}
        self.encoders = {}
        os.makedirs('artifacts/encode', exist_ok=True)
    
    def encode(self,df):
        ################# Pandas code ###############

        # for column in self.nominal_columns:
        #     unique_values = df[column].unique()
        #     encoder_dict = {value: i for i,value in enumerate(unique_values)}
        #     self.encoder_dicts[column]=encoder_dict

        #     encoder_path = os.path.join('artifacts/encode',f"{column}_encoder.json")
        #     with open(encoder_path,"w") as f:
        #         json.dump(encoder_dict,f)
            
        #     df[column]= df[column].map(encoder_dict)
        
        # return df

        ################## Pyspark code ###########

        df_encoded = df
        for column in self.nominal_columns:
            unique_values = df_encoded.select(column).distinct().count()
            indexer = StringIndexer(
                inputCol=column,
                outputCol=f"{column}_index"
            )
            indexer_model = indexer.fit(df_encoded)
            self.indexers[column] = indexer_model
            
            lables = indexer_model.labels
            encoder_dict = {label:idx for idx, label in enumerate(lables)}
            self.encoder_dicts[column] = encoder_dict

            df_encoded = indexer_model.transform(df_encoded)
        
        return df_encoded
    
    def get_encoder_dicts(self) -> Dict[str, Dict[str,int]]:
        return self.encoder_dicts
    
    def get_indexers(self) -> Dict[str,StringIndexer]:
        return self.indexers
    
class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self,ordinal_mappings, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.ordinal_mappings=ordinal_mappings
    
    def encode(self,df):
        ############### Pnadas code ######################

        # for column,mapping in self.ordinal_mappings.items():
        #     df[column]=df[column].map(mapping)
        #     logging.info(f"Encoded ordinal variable '{column}' with {len(mapping)} categories")
        
        # return df

        ############## Pyspark code #######################
        df_encoded = df

        for column, mapping in self.ordinal_mappings.items():
            mapping_expr = F.when(F.col(column).isNull(),None)
            for value, code in mapping.items():
                mapping_expr = mapping_expr.when(F.col(column)==value,code)
            
            df_encoded = df_encoded.withColumn(column,mapping_expr)
        
        return df_encoded