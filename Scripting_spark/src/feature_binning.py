import logging
import pandas as pd
from abc import ABC,abstractmethod
from typing import Dict,List,Optional,Union
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from spark_session import get_or_create_spark_session
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureBinningStrategy(ABC):

    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def bin_feature(self,df:pd.DataFrame,column:str)->pd.DataFrame:
        pass

class CustomBinningStrategy(FeatureBinningStrategy):
    def __init__(self, bin_definitions: Dict[str, List[float]], spark: Optional[SparkSession] = None):
       super().__init__(spark)
       self.bin_definitions = bin_definitions

    def bin_feature(self, df,column):
        ############################# Pandas Code ###############
        # def assign_value(value):
        #     if value == 850:
        #         return "Excellent"
            
        #     for bin_label,bin_range in self.bin_definition.items():
        #         if len(bin_range)==2:
        #             if(bin_range[0]) <= value <=bin_range[1]:
        #                 return bin_label
        #         elif len(bin_range) == 1:
        #             if value >= bin_range[0]:
        #                     return bin_label
        #     if value>850:
        #         return "Invalid"
        
        # df[f"{column}Bins"] = df[column].apply(assign_value)
        # del df[column]

        # return df

        ################################# Pyspark code ###########################
        bin_column = f'{column}Bins'

        case_expr = F.when(F.col(column)==850, "Excellent")

        for bin_label, bin_range in self.bin_definitions.items():
            if len(bin_range) == 2:
                case_expr = case_expr.when(
                    (F.col(column) >= bin_range[0]) & (F.col(column) <= bin_range[1]),
                    bin_label
                )
            elif len(bin_range) == 1:
                case_expr = case_expr.when(
                                        (F.col(column) >= bin_range[0]), 
                                        bin_label
                                        )
        
        df_binned = df.withColumn(bin_column,case_expr)
        df_binned = df_binned.drop(column)

        return df_binned
