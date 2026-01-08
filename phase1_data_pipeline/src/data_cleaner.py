"""
Data cleaning and validation module
"""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Clean and validate stock data
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_rows = len(df)
    
    def remove_duplicates(self) -> 'DataCleaner':
        """Remove duplicate rows"""
        before = len(self.df)
        self.df.drop_duplicates(subset=['date', 'ticker'], keep='first', inplace=True)
        after = len(self.df)
        
        if before != after:
            logger.info(f"Removed {before - after} duplicate rows")
        
        return self
    
    def handle_missing_values(self, method: str = 'drop') -> 'DataCleaner':
        """Handle missing values"""
        missing_pct = (self.df.isnull().sum() / len(self.df)) * 100
        
        if missing_pct.any():
            logger.info(f"Missing values:\n{missing_pct[missing_pct > 0]}")
        
        if method == 'drop':
            critical_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            before = len(self.df)
            self.df.dropna(subset=critical_columns, inplace=True)
            after = len(self.df)
            if before != after:
                logger.info(f"Dropped {before - after} rows with missing values")
        
        return self
    
    def remove_outliers(self, columns: list = None, std_threshold: float = 4) -> 'DataCleaner':
        """Remove statistical outliers"""
        if columns is None:
            columns = ['open', 'high', 'low', 'close', 'volume']
        
        before = len(self.df)
        
        for col in columns:
            if col in self.df.columns:
                mean = self.df[col].mean()
                std = self.df[col].std()
                z_scores = np.abs((self.df[col] - mean) / std)
                self.df = self.df[z_scores < std_threshold]
        
        after = len(self.df)
        if before != after:
            logger.info(f"Removed {before - after} outlier rows")
        
        return self
    
    def validate_data(self) -> Tuple[bool, list]:
        """Validate data quality"""
        issues = []
        
        if len(self.df) < 100:
            issues.append(f"Insufficient data: {len(self.df)} rows")
        
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (self.df[col] <= 0).any():
                issues.append(f"Invalid values in {col}")
        
        if (self.df['high'] < self.df['low']).any():
            issues.append("High < Low in some rows")
        
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        is_valid = len(issues) == 0
        
        if is_valid:
            logger.info("✓ Data validation passed!")
        else:
            logger.warning(f"✗ Validation issues: {issues}")
        
        return is_valid, issues
    
    def fix_data_types(self) -> 'DataCleaner':
        """Ensure correct data types"""
        self.df['date'] = pd.to_datetime(self.df['date'])
        
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        if 'ticker' in self.df.columns:
            self.df['ticker'] = self.df['ticker'].astype(str)
        
        return self
    
    def sort_by_date(self) -> 'DataCleaner':
        """Sort by date"""
        self.df.sort_values('date', inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self
    
    def clean_all(self) -> pd.DataFrame:
        """Apply all cleaning steps"""
        logger.info(f"Starting cleaning. Original rows: {self.original_rows}")
        
        self.fix_data_types()
        self.remove_duplicates()
        self.handle_missing_values()
        self.remove_outliers()
        self.sort_by_date()
        self.validate_data()
        
        logger.info(f"Cleaning complete. Final rows: {len(self.df)}")
        
        return self.df