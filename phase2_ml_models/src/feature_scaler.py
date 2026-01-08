"""
Feature scaling for ML models
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureScaler:
    """
    Scale features for machine learning
    """
    
    def __init__(self, method='minmax'):
        """
        Args:
            method: 'minmax' or 'standard'
        """
        self.method = method
        
        if method == 'minmax':
            self.scaler_X = MinMaxScaler(feature_range=(0, 1))
            self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler_X = StandardScaler()
            self.scaler_y = StandardScaler()
        
        logger.info(f"Initialized {method} scaler")
    
    def fit_transform(self, X_train, y_train, X_test=None, y_test=None):
        """
        Fit on training data and transform both train and test
        """
        # Scale features
        X_train_scaled = self.scaler_X.fit_transform(X_train)
        
        # Scale target
        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        result = {
            'X_train_scaled': X_train_scaled,
            'y_train_scaled': y_train_scaled
        }
        
        # Scale test data if provided
        if X_test is not None:
            X_test_scaled = self.scaler_X.transform(X_test)
            result['X_test_scaled'] = X_test_scaled
        
        if y_test is not None:
            y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()
            result['y_test_scaled'] = y_test_scaled
        
        logger.info("✓ Features scaled successfully")
        
        return result
    
    def inverse_transform_y(self, y_scaled):
        """Convert scaled predictions back to original scale"""
        return self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
    
    def inverse_transform_X(self, X_scaled):
        """Convert scaled features back to original scale"""
        return self.scaler_X.inverse_transform(X_scaled)
    
    def save(self, filepath: str):
        """Save scaler to disk"""
        joblib.dump({
            'scaler_X': self.scaler_X,
            'scaler_y': self.scaler_y,
            'method': self.method
        }, filepath)
        logger.info(f"Scaler saved to {filepath}")
    
    def load(self, filepath: str):
        """Load scaler from disk"""
        data = joblib.load(filepath)
        self.scaler_X = data['scaler_X']
        self.scaler_y = data['scaler_y']
        self.method = data['method']
        logger.info(f"Scaler loaded from {filepath}")


if __name__ == "__main__":
    # Test scaler
    from data_loader import DataLoader
    
    loader = DataLoader('AAPL')
    data_split = loader.train_test_split()
    
    scaler = FeatureScaler(method='minmax')
    scaled_data = scaler.fit_transform(
        data_split['X_train'],
        data_split['y_train'],
        data_split['X_test'],
        data_split['y_test']
    )
    
    print(f"\nOriginal X range: [{data_split['X_train'].min():.2f}, {data_split['X_train'].max():.2f}]")
    print(f"Scaled X range: [{scaled_data['X_train_scaled'].min():.2f}, {scaled_data['X_train_scaled'].max():.2f}]")
    
    print(f"\nOriginal y range: [{data_split['y_train'].min():.2f}, {data_split['y_train'].max():.2f}]")
    print(f"Scaled y range: [{scaled_data['y_train_scaled'].min():.2f}, {scaled_data['y_train_scaled'].max():.2f}]")