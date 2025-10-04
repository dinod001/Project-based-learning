import logging
import pandas as pd
from abc import ABC,abstractmethod
from typing import List,Optional,Dict,Tuple
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import BooleanType
from spark_session import get_or_create_spark_session
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(levelname)s - %(message)s')

class OutlierDetectionStrategy(ABC):

    def __init__(self, spark: Optional[SparkSession] = None):
        """Initialize with SparkSession."""
        self.spark = spark or get_or_create_spark_session()

    @abstractmethod
    def detect_coutliers(self,df:pd.DataFrame,columns: list) -> pd.DataFrame:
        pass

    @abstractmethod
    def get_outlier_bounds(self, df: DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Get outlier bounds for specified columns.
        
        Args:
            df: DataFrame (PySpark or pandas)
            columns: List of column names
            
        Returns:
            Dictionary mapping column names to (lower_bound, upper_bound) tuples
        """
        pass

class IQROutlierDetection(OutlierDetectionStrategy):
    def __init__(self,threshold: float = 1.5, spark: Optional[SparkSession]=None):
        """
        Initialize IQR outlier detection.
        
        Args:
            threshold: IQR multiplier for outlier bounds (default: 1.5)
            spark: Optional SparkSession
        """
        super().__init__(spark)
        self.threshold = threshold
    
    def get_outlier_bounds(self, df: DataFrame, columns: List[str]) -> Dict[str, Tuple[float, float]]:
        """
        Calculate outlier bounds using IQR method.
        
        Args:
            df: PySpark DataFrame
            columns: List of column names
            
        Returns:
            Dictionary mapping column names to (lower_bound, upper_bound) tuples
        """
        bounds={}

        for col in columns:
            ############### PANDAS CODES ###########################
            # Q1 = df[col].quantile(0.25)
            # Q3 = df[col].quantile(0.75)
            # IQR = Q3 - Q1
            
            # lower_bound = Q1 - self.threshold * IQR
            # upper_bound = Q3 + self.threshold * IQR
            
            # bounds[col] = (lower_bound, upper_bound)

            ############### PYSPARK CODES ###########################
            quantiles = df.approxQuantile(self.relevant_column,[0.25,0.75],0.01)
            Q1,Q3  = quantiles[0], quantiles[1]
            IQR = Q3 - Q1

            lower_bound = Q1 - self.threshold * IQR
            upper_bound = Q3 + self.threshold * IQR

            bounds[col] = (lower_bound, upper_bound)
        
        return bounds


    def detect_coutliers(self,df,columns):
         # Get outlier bounds
        bounds = self.get_outlier_bounds(df, columns)

        # Add outlier indicator columns
        result_df = df
        total_outliers = 0

        for col in columns:
            ############### PANDAS CODES ###########################
            # df[col] = df[col].astype(float)
            
            ############### PYSPARK CODES ###########################
            # PySpark handles type conversions automatically
            
            lower_bound, upper_bound = bounds[col]
            
            # Create outlier indicator column
            outlier_col = f"{col}_outlier"
            
            ############### PANDAS CODES ###########################
            # outliers[col] = (df[col] < lower_bound) | (df[col] > upper_bound)
            # outlier_count = outliers[col].sum()
            # total_rows = len(df)
            
            ############### PYSPARK CODES ###########################
            result_df = result_df.withColumn(
                                            outlier_col,
                                            (F.col(col) < lower_bound) | (F.col(col) > upper_bound)
                                            )

            outlier_count = result_df.filter(F.col(outlier_col)).count()
            total_rows = df.count()

            return result_df

class OutlierDetector:
    def __init__(self,strategy):
        self._strategy = strategy
    
    def detect_coutliers(self,df,selected_columns):
        return self._strategy.detect_coutliers(df,selected_columns)
    
    def handle_outliers(self, df: DataFrame, selected_columns: List[str], 
                       method: str = 'remove', min_outliers: int = 2) -> DataFrame:
        ############### PANDAS CODES ###########################
        # initial_rows = len(df)
        
        ############### PYSPARK CODES ###########################
        initial_rows = df.count()

        if method == 'remove':
            # Add outlier indicator columns
            df_with_outliers = self.detect_outliers(df, selected_columns)
            
            # Count outliers per row
            outlier_columns = [f"{col}_outlier" for col in selected_columns]
            
            ############### PANDAS CODES ###########################
            # outlier_count = outliers.sum(axis=1)
            # rows_to_remove = outlier_count >= min_outliers
            # cleaned_df = df[~rows_to_remove]
            
            ############### PYSPARK CODES ###########################
            outlier_count_expr = sum(F.col(col).cast('int') for col in outlier_columns)
            df_with_count = df_with_outliers.withColumn('outlier_count', outlier_count_expr)
            clean_df = df_with_count.filter(F.col("outlier_count") < min_outliers)
            clean_df = clean_df.drop("outlier_count")
            
            ############### PANDAS CODES ###########################
            # rows_removed = rows_to_remove.sum()
            
            ############### PYSPARK CODES ###########################
            rows_removed = initial_rows - clean_df.count()
            
        elif method == 'cap':
            bounds = self._strategy.get_outlier_bounds(df, selected_columns)
            clean_df = df

            for col in selected_columns:
                lb, ub = bounds[col]

                clean_df = clean_df.withColumn(
                                            col,
                                            F.when(F.col(col) < lb, lb)
                                            .when(F.col(col) > ub, ub)
                                            .otherwise(F.col(col))
                                            )
            
        else:
            raise ValueError(f"Unknown outlier handling method: {method}")
        

        return clean_df


    
