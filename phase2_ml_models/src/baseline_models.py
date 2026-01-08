"""
Baseline ML Models: Linear Regression, Random Forest
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaselineModels:
    """
    Baseline machine learning models
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Args:
            model_type: 'linear_regression' or 'random_forest'
        """
        self.model_type = model_type
        
        if model_type == 'linear_regression':
            self.model = LinearRegression()
            logger.info("Initialized Linear Regression")
        
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            logger.info("Initialized Random Forest")
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train(self, X_train, y_train):
        """Train the model"""
        logger.info(f"Training {self.model_type}...")
        
        self.model.fit(X_train, y_train)
        
        # Training score
        train_pred = self.model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        
        logger.info(f"✓ Training complete. Train RMSE: {train_rmse:.4f}")
        
        return self
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        # Directional Accuracy
        actual_direction = np.diff(y_test) > 0
        pred_direction = np.diff(predictions) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
        
        logger.info(f"\n{'='*60}")
        logger.info(f"{self.model_type.upper()} - Evaluation Metrics")
        logger.info(f"{'='*60}")
        for metric, value in metrics.items():
            logger.info(f"{metric:<25}: {value:.4f}")
        logger.info(f"{'='*60}\n")
        
        return metrics, predictions
    
    def save(self, filepath: str):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    # Test baseline models
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from data_loader import DataLoader
    from feature_scaler import FeatureScaler
    
    # Load data
    loader = DataLoader('AAPL')
    data_split = loader.train_test_split()
    
    # Scale data
    scaler = FeatureScaler()
    scaled_data = scaler.fit_transform(
        data_split['X_train'],
        data_split['y_train'],
        data_split['X_test'],
        data_split['y_test']
    )
    
    # Test Linear Regression
    print("\n" + "="*60)
    print("TESTING LINEAR REGRESSION")
    print("="*60)
    
    lr_model = BaselineModels('linear_regression')
    lr_model.train(scaled_data['X_train_scaled'], scaled_data['y_train_scaled'])
    lr_metrics, lr_pred = lr_model.evaluate(scaled_data['X_test_scaled'], scaled_data['y_test_scaled'])
    
    # Test Random Forest
    print("\n" + "="*60)
    print("TESTING RANDOM FOREST")
    print("="*60)
    
    rf_model = BaselineModels('random_forest')
    rf_model.train(scaled_data['X_train_scaled'], scaled_data['y_train_scaled'])
    rf_metrics, rf_pred = rf_model.evaluate(scaled_data['X_test_scaled'], scaled_data['y_test_scaled'])