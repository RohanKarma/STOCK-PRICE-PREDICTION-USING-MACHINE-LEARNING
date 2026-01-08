"""
Make future predictions using trained models
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockPredictor:
    """
    Make future stock price predictions
    """
    
    def __init__(self, model, scaler, sequence_creator=None):
        """
        Args:
            model: Trained model
            scaler: FeatureScaler object
            sequence_creator: SequenceCreator object (for LSTM)
        """
        self.model = model
        self.scaler = scaler
        self.sequence_creator = sequence_creator
    
    def predict_next_day(self, latest_data):
        """
        Predict the next day's price
        
        Args:
            latest_data: Latest features (for LSTM: sequence_length days)
        
        Returns:
            predicted_price: Predicted closing price
        """
        # Scale the data
        if self.sequence_creator is not None:
            # LSTM prediction
            scaled_data = self.scaler.scaler_X.transform(latest_data)
            sequence = self.sequence_creator.create_future_sequence(scaled_data)
            scaled_prediction = self.model.predict(sequence)[0]
        else:
            # Traditional ML prediction
            latest_features = latest_data[-1:].reshape(1, -1)
            scaled_features = self.scaler.scaler_X.transform(latest_features)
            scaled_prediction = self.model.predict(scaled_features)[0]
        
        # Inverse transform to get actual price
        predicted_price = self.scaler.inverse_transform_y(np.array([scaled_prediction]))[0]
        
        return predicted_price
    
    def predict_future(self, latest_data, n_days=30):
        """
        Predict multiple days into the future
        
        Args:
            latest_data: Latest features
            n_days: Number of days to predict
        
        Returns:
            predictions: Array of predicted prices
            dates: Array of prediction dates
        """
        logger.info(f"Predicting next {n_days} days...")
        
        predictions = []
        current_data = latest_data.copy()
        
        for day in range(n_days):
            # Predict next day
            next_price = self.predict_next_day(current_data)
            predictions.append(next_price)
            
            # Update data for next prediction
            # (Simplified: in practice, you'd need to update all features)
            if len(current_data) > 1:
                current_data = np.roll(current_data, -1, axis=0)
                # This is a simplification - ideally update all features
        
        # Generate future dates
        last_date = datetime.now()
        future_dates = [last_date + timedelta(days=i+1) for i in range(n_days)]
        
        logger.info(f"✓ Generated {n_days} predictions")
        
        return np.array(predictions), future_dates
    
    def predict_with_confidence(self, latest_data, n_days=30, confidence=0.95):
        """
        Predict with confidence intervals
        
        Args:
            latest_data: Latest features
            n_days: Number of days to predict
            confidence: Confidence level (e.g., 0.95 for 95%)
        
        Returns:
            predictions: Dict with 'mean', 'lower', 'upper' bounds
            dates: Prediction dates
        """
        # Get base prediction
        mean_predictions, dates = self.predict_future(latest_data, n_days)
        
        # Calculate confidence intervals (simplified approach)
        # In practice, use bootstrap or model uncertainty
        std_dev = np.std(mean_predictions) * 0.1  # Simplified
        z_score = 1.96 if confidence == 0.95 else 2.576  # 95% or 99%
        
        margin = z_score * std_dev
        
        lower_bound = mean_predictions - margin
        upper_bound = mean_predictions + margin
        
        predictions = {
            'mean': mean_predictions,
            'lower': lower_bound,
            'upper': upper_bound,
            'confidence': confidence
        }
        
        return predictions, dates
    
    def create_prediction_df(self, predictions, dates, ticker):
        """
        Create a DataFrame of predictions
        
        Args:
            predictions: Array or dict of predictions
            dates: Array of dates
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame with predictions
        """
        if isinstance(predictions, dict):
            df = pd.DataFrame({
                'date': dates,
                'ticker': ticker,
                'predicted_price': predictions['mean'],
                'lower_bound': predictions['lower'],
                'upper_bound': predictions['upper']
            })
        else:
            df = pd.DataFrame({
                'date': dates,
                'ticker': ticker,
                'predicted_price': predictions
            })
        
        return df


if __name__ == "__main__":
    # Test predictor
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from data_loader import DataLoader
    from feature_scaler import FeatureScaler
    from baseline_models import BaselineModels
    
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
    
    # Train a simple model
    model = BaselineModels('random_forest')
    model.train(scaled_data['X_train_scaled'], scaled_data['y_train_scaled'])
    
    # Create predictor
    predictor = StockPredictor(model, scaler)
    
    # Get latest data
    latest_data = loader.get_latest_data(n_days=60)
    
    # Make predictions
    predictions, dates = predictor.predict_future(latest_data, n_days=30)
    
    print("\nPredictions for next 30 days:")
    for date, price in zip(dates[:5], predictions[:5]):
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")