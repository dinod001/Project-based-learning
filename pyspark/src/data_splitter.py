"""
Data splitting strategies for both pandas and PySpark DataFrames.
Students can compare train-test split implementations between pandas and PySpark.
"""

import logging
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import pandas as pd  # Keep for educational comparison
from sklearn.model_selection import train_test_split  # For pandas implementation
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from spark_session import get_or_create_spark_session

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplittingStrategy(ABC):
    """Abstract base class for data splitting strategies."""
    ############### PYSPARK CODES ###########################
    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()
    
    @abstractmethod
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Split data into train and test sets.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        pass


class SimpleTrainTestSplitStrategy(DataSplittingStrategy):
    """Simple random train-test split strategy."""
    
    ############### PYSPARK CODES ###########################
    def __init__(self, test_size: float = 0.2, random_seed: int = 42, spark: Optional[SparkSession] = None):
        """
        Initialize simple train-test split strategy.
        
        Args:
            test_size: Proportion of data for test set (0-1)
            random_seed: Random seed for reproducibility
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.test_size = test_size
        self.train_size = 1.0 - test_size
        self.random_seed = random_seed
        logger.info(f"SimpleTrainTestSplitStratergy initialized with test_size={test_size}")
    
    def split_data(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Perform simple random train-test split.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        pass


class DataSplitter:
    """Main data splitter class that uses different strategies."""
    
    def __init__(self, strategy: DataSplittingStrategy):
        """
        Initialize data splitter with a specific strategy.
        
        Args:
            strategy: DataSplittingStrategy instance
        """
        self.strategy = strategy
        logger.info(f"DataSplitter initialized with strategy: {strategy.__class__.__name__}")
    
    def split(self, df: DataFrame, target_column: str) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
        """
        Split data using the configured strategy.
        
        Args:
            df: PySpark DataFrame
            target_column: Name of the target column
            
        Returns:
            Tuple of (X_train, X_test, Y_train, Y_test) DataFrames
        """
        return self.strategy.split_data(df, target_column)

def create_simple_splitter(test_size: float = 0.2, spark: Optional[SparkSession] = None):
    """Create a simple train-test splitter (backward compatibility)."""
    return SimpleTrainTestSplitStrategy(test_size=test_size, spark=spark)