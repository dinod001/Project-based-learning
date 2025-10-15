import logging
import os
import pandas as pd
import json
from enum import Enum
from abc import ABC,abstractmethod
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeatureEncodingStrategy(ABC):
    @abstractmethod
    def encode(self, df: pd.DataFrame) ->pd.DataFrame:
        pass

class NominalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self,nominal_columns):
        self.nominal_columns = nominal_columns
    
    def encode(self,df):
        for column in self.nominal_columns:
            df_dummies = pd.get_dummies(df[column],prefix=column)
            df = pd.concat([df,df_dummies],axis=1)
            del df[column]

        return df

class OrdinalEncodingStrategy(FeatureEncodingStrategy):
    def __init__(self, ordinal_mappings):
        self.ordinal_mappings = ordinal_mappings
    
    def encode(self,df):
        for column,mapping in self.ordinal_mappings.items():
            df[column] = df[column].map(mapping)
        
        return df
    
