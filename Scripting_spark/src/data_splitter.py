from typing import Tuple, Optional
from pyspark.sql import DataFrame, SparkSession
from pyspark.ml import PipelineModel
from abc import ABC, abstractmethod
from spark_session import get_or_create_spark_session  # your helper to get Spark session

class DataSplittingStrategy(ABC):
    """Abstract base class for data splitting strategies."""
    
    def __init__(self, spark: Optional[SparkSession] = None):
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Simple random train-test split strategy."""
    
    def __init__(self, test_size: float = 0.2, random_seed: int = 42, spark: Optional[SparkSession] = None):
        super().__init__(spark)
        self.test_size = test_size
        self.train_size = 1.0 - test_size
        self.random_seed = random_seed
    
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        # Randomly split DataFrame
        train_df, test_df = df.randomSplit([self.train_size, self.test_size], seed=self.random_seed)
        
        # Separate features and target
        X_train = train_df.drop(target_column)
        Y_train = train_df.select(target_column)
        
        X_test = test_df.drop(target_column)
        Y_test = test_df.select(target_column)
        
        return X_train, X_test, Y_train, Y_test


class DataSplitter:
    """Main data splitter class that uses a strategy."""
    
    def __init__(self, strategy: DataSplittingStrategy):
        self.strategy = strategy
    
    def split(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        return self.strategy.split_data(df, target_column)


def create_simple_splitter(test_size: float = 0.2, spark: Optional[SparkSession] = None):
    return SimpleTrainTestSplitStrategy(test_size=test_size, spark=spark)
