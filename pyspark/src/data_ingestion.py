import os
import pandas as pd
from abc import ABC,abstractmethod
from typing import Optional
from pyspark.sql import DataFrame, SparkSession
from spark_session import get_or_create_spark_session

class dataIngestor(ABC):
    #getting spark session
    def __init__(self,spark:Optional[SparkSession]=None):
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def ingest(self,file_path_or_link:str) -> pd.DataFrame:
        pass

class DataIngestorCSV(dataIngestor):
    # cv load by pandas

    # def ingest(self, file_path_or_link):
    #     return pd.read_csv(file_path_or_link)


    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
    #data load by using spark
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
    # data load by pandas 

    # def ingest(self, file_path_or_link):
    #     return pd.read_excel(file_path_or_link)

    # using spark
    def ingest(self, file_path_or_link: str, sheet_name: Optional[str] = None, **options) -> DataFrame:
        try:
            pandas_df = pd.read_csv(file_path_or_link)
            df = self.spark.createDataFrame(pandas_df)
        
        except Exception as e:
            raise

class DataIngestorParquet(dataIngestor):
    
    def ingest(self, file_path_or_link: str, **options) -> DataFrame:
        try:
            # Read Parquet file(s)
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

            df = self.spark.read.options(**csv_options).parquet(file_path_or_link)
            
        except Exception as e:
            raise

class DataIngestorFactory:
    @staticmethod
    def get_ingestor(file_path:str, spark:Optional[SparkSession]=None)-> dataIngestor:

        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.csv':
            return DataIngestorCSV(spark)
        elif file_extension in ['.xlsx', '.xls']:
            return DataIngestorExcel(spark)
        elif file_extension == '.parquet':
            return DataIngestorParquet(spark)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")