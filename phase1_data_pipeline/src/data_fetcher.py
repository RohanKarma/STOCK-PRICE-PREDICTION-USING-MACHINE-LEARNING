"""
Module for fetching stock data from Yahoo Finance
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class StockDataFetcher:
    """
    Fetches stock data from Yahoo Finance
    """
    
    def __init__(self):
        self.data_source = "Yahoo Finance"
    
    def get_stock_data(
        self, 
        ticker: str, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Fetch OHLCV data for a given ticker
        
        Args:
            ticker (str): Stock symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            interval (str): Data interval ('1d', '1h', etc.)
        
        Returns:
            pd.DataFrame: DataFrame with OHLCV data
        """
        try:
            if not end_date:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            if not start_date:
                start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
            
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
            
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {ticker}")
                return pd.DataFrame()
            
            df.reset_index(inplace=True)
            df.columns = df.columns.str.lower()
            df['ticker'] = ticker
            df = df[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume']]
            
            if df['date'].dt.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            
            logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(
        self, 
        tickers: List[str], 
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Fetch data for multiple tickers
        """
        all_data = []
        
        for ticker in tickers:
            df = self.get_stock_data(ticker, start_date, end_date)
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Successfully fetched data for {len(tickers)} tickers")
            return combined_df
        else:
            logger.warning("No data fetched for any ticker")
            return pd.DataFrame()
    
    def get_company_info(self, ticker: str) -> dict:
        """
        Get company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            company_data = {
                'ticker': ticker,
                'company_name': info.get('longName', 'N/A'),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'USD')
            }
            
            return company_data
            
        except Exception as e:
            logger.error(f"Error fetching company info for {ticker}: {str(e)}")
            return {}


if __name__ == "__main__":
    fetcher = StockDataFetcher()
    df = fetcher.get_stock_data('AAPL')
    print(f"\nFetched {len(df)} rows for AAPL")
    print(df.head())