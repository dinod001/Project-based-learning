import os
import pandas as pd
from abc import ABC,abstractmethod
from typing import Optional
from pyspark.sql import DataFrame,SparkSession
from spark_session import get_or_create_spark_session


class dataIngestor(ABC):

    def __init__(self,spark: Optional[SparkSession]=None) -> None:
        
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def ingest(self, file_path_or_link:str)->pd.DataFrame:
        pass

class DataIngestorCSV(dataIngestor):
    def ingest(self, file_path_or_link:str, **options):
        try:
            # Default CSV options
            csv_options = {
                        "header": "true",
                        "inferSchema": "true",
                        "ignoreLeadingWhiteSpace": "true",
                        "ignoreTrailingWhiteSpace": "true",
                        "nullValue": "",
                        "nanValue": "NaN",
                        "escape": '"',
                        "quote": '"'
                        }
            csv_options.update(options)

            df = self.spark.read.options(**csv_options).csv(file_path_or_link)
        
        except Exception as e:
            raise


class DataIngestorExcel(dataIngestor):
    def ingest(self, file_path_or_link:str):
        try:
            pandas_df = pd.read_excel(file_path_or_link)
            df = self.spark.createDataFrame(pandas_df)
        
        except Exception as e:
            raise

class DataIngestorParquet(dataIngestor):
    def ingest(self, file_path_or_link:str, **options):
        try:
            # Default CSV options
            csv_options = {
                        "header": "true",
                        "inferSchema": "true",
                        "ignoreLeadingWhiteSpace": "true",
                        "ignoreTrailingWhiteSpace": "true",
                        "nullValue": "",
                        "nanValue": "NaN",
                        "escape": '"',
                        "quote": '"'
                        }
            csv_options.update(options)

            df = self.spark.read.options(**csv_options).parquet(file_path_or_link)
        
        except Exception as e:
            raise

class DataIngestorFactory:
    """Factory class to create appropriate data ingestor based on file type."""
    
    @staticmethod
    def get_ingestor(file_path: str, spark: Optional[SparkSession] = None) -> DataIngestor:
        """
        Get appropriate data ingestor based on file extension.
        
        Args:
            file_path: Path to the data file
            spark: Optional SparkSession
            
        Returns:
            DataIngestor: Appropriate ingestor instance
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.csv':
            return DataIngestorCSV(spark)
        elif file_extension in ['.xlsx', '.xls']:
            return DataIngestorExcel(spark)
        elif file_extension == '.parquet':
            return DataIngestorParquet(spark)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")