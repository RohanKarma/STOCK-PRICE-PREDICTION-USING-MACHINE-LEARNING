"""
Feature engineering module - Calculate technical indicators
"""
import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Calculate technical indicators and create features for ML
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.df.sort_values('date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
    
    def add_moving_averages(self, periods: list = [20, 50, 200]) -> 'FeatureEngineer':
        """Add Simple Moving Averages (SMA) and Exponential Moving Averages (EMA)"""
        logger.info("Calculating Moving Averages...")
        
        for period in periods:
            self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
        
        return self
    
    def add_rsi(self, period: int = 14) -> 'FeatureEngineer':
        """Add Relative Strength Index (RSI)"""
        logger.info("Calculating RSI...")
        
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        return self
    
    def add_macd(self, fast: int = 12, slow: int = 26, signal: int = 9) -> 'FeatureEngineer':
        """Add MACD (Moving Average Convergence Divergence)"""
        logger.info("Calculating MACD...")
        
        ema_fast = self.df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = self.df['close'].ewm(span=slow, adjust=False).mean()
        
        self.df['macd'] = ema_fast - ema_slow
        self.df['macd_signal'] = self.df['macd'].ewm(span=signal, adjust=False).mean()
        self.df['macd_histogram'] = self.df['macd'] - self.df['macd_signal']
        
        return self
    
    def add_bollinger_bands(self, period: int = 20, std_dev: int = 2) -> 'FeatureEngineer':
        """Add Bollinger Bands"""
        logger.info("Calculating Bollinger Bands...")
        
        self.df['bb_middle'] = self.df['close'].rolling(window=period).mean()
        std = self.df['close'].rolling(window=period).std()
        
        self.df['bb_upper'] = self.df['bb_middle'] + (std_dev * std)
        self.df['bb_lower'] = self.df['bb_middle'] - (std_dev * std)
        self.df['bb_width'] = self.df['bb_upper'] - self.df['bb_lower']
        self.df['bb_percent'] = (self.df['close'] - self.df['bb_lower']) / (self.df['bb_upper'] - self.df['bb_lower'])
        
        return self
    
    def add_atr(self, period: int = 14) -> 'FeatureEngineer':
        """Add Average True Range (ATR)"""
        logger.info("Calculating ATR...")
        
        high_low = self.df['high'] - self.df['low']
        high_close = np.abs(self.df['high'] - self.df['close'].shift())
        low_close = np.abs(self.df['low'] - self.df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.df['atr'] = true_range.rolling(window=period).mean()
        
        return self
    
    def add_volatility(self, window: int = 30) -> 'FeatureEngineer':
        """Add historical volatility"""
        logger.info("Calculating Volatility...")
        
        self.df['returns'] = self.df['close'].pct_change()
        self.df['volatility'] = self.df['returns'].rolling(window=window).std() * np.sqrt(252)
        
        return self
    
    def add_volume_indicators(self) -> 'FeatureEngineer':
        """Add volume-based indicators"""
        logger.info("Calculating Volume Indicators...")
        
        self.df['volume_sma_20'] = self.df['volume'].rolling(window=20).mean()
        self.df['volume_ratio'] = self.df['volume'] / self.df['volume_sma_20']
        
        # On-Balance Volume (OBV)
        obv = []
        obv_value = 0
        
        for i in range(len(self.df)):
            if i == 0:
                obv.append(0)
            else:
                if self.df['close'].iloc[i] > self.df['close'].iloc[i-1]:
                    obv_value += self.df['volume'].iloc[i]
                elif self.df['close'].iloc[i] < self.df['close'].iloc[i-1]:
                    obv_value -= self.df['volume'].iloc[i]
                obv.append(obv_value)
        
        self.df['obv'] = obv
        
        return self
    
    def add_price_features(self) -> 'FeatureEngineer':
        """Add price-based features"""
        logger.info("Calculating Price Features...")
        
        self.df['price_change'] = self.df['close'].diff()
        self.df['price_change_pct'] = self.df['close'].pct_change() * 100
        self.df['daily_range'] = self.df['high'] - self.df['low']
        self.df['daily_range_pct'] = (self.df['daily_range'] / self.df['close']) * 100
        self.df['gap'] = self.df['open'] - self.df['close'].shift(1)
        self.df['gap_pct'] = (self.df['gap'] / self.df['close'].shift(1)) * 100
        
        return self
    
    def add_momentum_indicators(self) -> 'FeatureEngineer':
        """Add momentum indicators"""
        logger.info("Calculating Momentum Indicators...")
        
        # Rate of Change (ROC)
        period = 12
        self.df['roc'] = ((self.df['close'] - self.df['close'].shift(period)) / 
                          self.df['close'].shift(period)) * 100
        
        # Stochastic Oscillator
        period = 14
        low_min = self.df['low'].rolling(window=period).min()
        high_max = self.df['high'].rolling(window=period).max()
        
        self.df['stoch_k'] = 100 * ((self.df['close'] - low_min) / (high_max - low_min))
        self.df['stoch_d'] = self.df['stoch_k'].rolling(window=3).mean()
        
        return self
    
    def add_all_features(self) -> pd.DataFrame:
        """Add all technical indicators at once"""
        logger.info("Adding ALL technical indicators...")
        
        self.add_moving_averages([20, 50, 200])
        self.add_rsi(14)
        self.add_macd()
        self.add_bollinger_bands()
        self.add_atr()
        self.add_volatility()
        self.add_volume_indicators()
        self.add_price_features()
        self.add_momentum_indicators()
        
        logger.info(f"Feature engineering complete. Total columns: {len(self.df.columns)}")
        
        return self.df
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """Return the DataFrame with all features"""
        return self.df


if __name__ == "__main__":
    from data_fetcher import StockDataFetcher
    
    fetcher = StockDataFetcher()
    df = fetcher.get_stock_data('AAPL')
    
    engineer = FeatureEngineer(df)
    df_with_features = engineer.add_all_features()
    
    print(f"\nColumns after feature engineering: {len(df_with_features.columns)}")
    print(df_with_features.tail())