"""
Database operations for storing stock data
"""
import pandas as pd
from sqlalchemy import create_engine
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """
    Manage database operations
    """
    
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        logger.info(f"Database initialized")
    
    def save_dataframe(self, df: pd.DataFrame, table_name: str = 'stock_data', 
                      if_exists: str = 'append') -> None:
        """Save DataFrame to database"""
        try:
            df.to_sql(table_name, self.engine, if_exists=if_exists, index=False)
            logger.info(f"Saved {len(df)} rows to {table_name}")
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            raise
    
    def load_dataframe(self, ticker: str = None, table_name: str = 'stock_data') -> pd.DataFrame:
        """Load data from database"""
        try:
            query = f"SELECT * FROM {table_name}"
            
            if ticker:
                query += f" WHERE ticker = '{ticker}'"
            
            query += " ORDER BY date"
            
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} rows from database")
            
            return df
        except Exception as e:
            logger.error(f"Error loading from database: {str(e)}")
            return pd.DataFrame()
    
    def get_available_tickers(self, table_name: str = 'stock_data') -> list:
        """Get list of available tickers"""
        try:
            query = f"SELECT DISTINCT ticker FROM {table_name}"
            df = pd.read_sql(query, self.engine)
            return df['ticker'].tolist()
        except:
            return []