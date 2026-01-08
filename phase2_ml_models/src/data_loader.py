"""
Load and prepare data for machine learning
"""
import pandas as pd
import numpy as np
import os
import sys
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config.config import PHASE1_DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_SIZE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and prepare stock data for ML
    """
    
    def __init__(self, ticker: str):
        self.ticker = ticker
        self.data = None
        self.train_data = None
        self.test_data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load processed data from Phase 1"""
        try:
            filepath = os.path.join(PHASE1_DATA_DIR, f'{self.ticker}_processed.csv')
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found: {filepath}")
            
            self.data = pd.read_csv(filepath)
            self.data['date'] = pd.to_datetime(self.data['date'])
            self.data.sort_values('date', inplace=True)
            self.data.reset_index(drop=True, inplace=True)
            
            logger.info(f"✓ Loaded {len(self.data)} rows for {self.ticker}")
            
            return self.data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def prepare_features(self) -> tuple:
        """
        Prepare features and target
        
        Returns:
            X (features), y (target)
        """
        if self.data is None:
            self.load_data()
        
        # Remove rows with NaN values
        df_clean = self.data.dropna()
        
        # Select features
        available_features = [col for col in FEATURE_COLUMNS if col in df_clean.columns]
        
        if len(available_features) < len(FEATURE_COLUMNS):
            missing = set(FEATURE_COLUMNS) - set(available_features)
            logger.warning(f"Missing features: {missing}")
        
        X = df_clean[available_features].values
        y = df_clean[TARGET_COLUMN].values
        dates = df_clean['date'].values
        
        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Target shape: {y.shape}")
        logger.info(f"Using {len(available_features)} features")
        
        return X, y, dates, available_features
    
    def train_test_split(self, test_size: float = None):
        """
        Split data into train and test sets (time-series aware)
        """
        if test_size is None:
            test_size = 1 - TRAIN_SIZE
        
        X, y, dates, features = self.prepare_features()
        
        # Calculate split index
        split_idx = int(len(X) * (1 - test_size))
        
        # Split data
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        dates_train, dates_test = dates[:split_idx], dates[split_idx:]
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Train/Test Split for {self.ticker}")
        logger.info(f"{'='*60}")
        logger.info(f"Training set:   {len(X_train)} samples ({dates_train[0]} to {dates_train[-1]})")
        logger.info(f"Testing set:    {len(X_test)} samples ({dates_test[0]} to {dates_test[-1]})")
        logger.info(f"Split ratio:    {(1-test_size)*100:.0f}% / {test_size*100:.0f}%")
        logger.info(f"{'='*60}\n")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'dates_train': dates_train,
            'dates_test': dates_test,
            'features': features
        }
    
    def get_latest_data(self, n_days: int = 60) -> np.ndarray:
        """Get the latest n days of data for prediction"""
        if self.data is None:
            self.load_data()
        
        df_clean = self.data.dropna()
        available_features = [col for col in FEATURE_COLUMNS if col in df_clean.columns]
        
        latest = df_clean[available_features].tail(n_days).values
        
        return latest


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader('AAPL')
    loader.load_data()
    
    # Test train/test split
    data_split = loader.train_test_split()
    
    print(f"\nTrain features shape: {data_split['X_train'].shape}")
    print(f"Test features shape: {data_split['X_test'].shape}")