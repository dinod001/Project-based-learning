import logging
import pandas as pd
from enum import Enum
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE  # ✅ Added import

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DataSplittingStrategy(ABC):
    @abstractmethod
    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        pass


class SplitType(str, Enum):
    SIMPLE = 'simple'
    STRATIFIED = 'stratified'


class SimpleTrainTestSplitStratergy(DataSplittingStrategy):
    def __init__(self, test_size=0.2, apply_smote=False):
        self.test_size = test_size
        self.apply_smote = apply_smote  # ✅ flag to control SMOTE usage

    def split_data(self, df: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        Y = df[target_column]
        X = df.drop(columns=[target_column])

        # Split data
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=self.test_size, random_state=42, stratify=Y
        )

        logger.info(f"Train-Test Split done: Train={X_train.shape}, Test={X_test.shape}")

        # ✅ Apply SMOTE only on the training set (to avoid data leakage)
        if self.apply_smote:
            logger.info("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=42)
            X_train, Y_train = smote.fit_resample(X_train, Y_train)
            logger.info(f"After SMOTE: X_train={X_train.shape}, Class distribution={Y_train.value_counts().to_dict()}")

        return X_train, X_test, Y_train, Y_test
