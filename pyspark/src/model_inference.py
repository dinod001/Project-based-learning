import json
import os
import sys
import joblib
from typing import Any, Dict
import pandas as pd
from feature_binning import CustomBinningStrategy
from feature_encoding import OrdinalEncodingStrategy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from config import get_binning_config, get_encoding_config

class ModelInference:
    
    def __init__(self, model_path: str):
        if not model_path or not isinstance(model_path, str):
            raise ValueError("Invalid model path provided")
        self.model_path = model_path
        self.encoders = {}
        self.model = None
        self.load_model()
        self.binning_config = get_binning_config()
        self.encoding_config = get_encoding_config()

    def load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        self.model = joblib.load(self.model_path)

    def load_encoders(self, encoders_dir: str) -> None:
        if not os.path.exists(encoders_dir):
            raise FileNotFoundError(f"Encoders directory not found: {encoders_dir}")
        encoder_files = [f for f in os.listdir(encoders_dir) if f.endswith('_encoder.json')]
        for file in encoder_files:
            feature_name = file.split('_encoder.json')[0]
            file_path = os.path.join(encoders_dir, file)
            with open(file_path, 'r') as f:
                self.encoders[feature_name] = json.load(f)

    def preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        if not data or not isinstance(data, dict):
            raise ValueError("Input data must be a non-empty dictionary")
        df = pd.DataFrame([data])

        # Apply encoders
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[col] = df[col].map(encoder)

        # Feature binning
        if 'CreditScore' in df.columns:
            binning = CustomBinningStrategy(self.binning_config['credit_score_bins'])
            df = binning.bin_feature(df, 'CreditScore')

        # Ordinal encoding
        ordinal_strategy = OrdinalEncodingStrategy(self.encoding_config['ordinal_mappings'])
        df = ordinal_strategy.encode(df)

        # Drop unnecessary columns
        drop_columns = ['RowNumber', 'CustomerId', 'Firstname', 'Lastname']
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])
        return df

    def predict(self, data: Dict[str, Any]) -> Dict[str, str]:
        if not data:
            raise ValueError("Input data cannot be empty")
        if self.model is None:
            raise ValueError("Model not loaded")
        processed_data = self.preprocess_input(data)
        y_pred = self.model.predict(processed_data)
        y_proba = self.model.predict_proba(processed_data)[:, 1]
        prediction = int(y_pred[0])
        probability = float(y_proba[0])
        status = 'Churn' if prediction == 1 else 'Retain'
        confidence = round(probability * 100, 2)
        return {"Status": status, "Confidence": f"{confidence}%"}
