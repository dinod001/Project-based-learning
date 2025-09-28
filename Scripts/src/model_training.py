import os
import joblib
import logging
import time
from typing import Any, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def train(
            self,
            model,
            X_train,
            Y_train
            ):
        model.fit(X_train,Y_train)
        train_score = model.score(X_train,Y_train)
        return model,train_score
    
    def save_model(self,model,filepath):  
        joblib.dump(model,filepath)
    
    def load(self,filepath):
        if not os.path.exists(filepath):
            raise ValueError("Can't load. File not found.")
        
        return joblib.load(filepath)