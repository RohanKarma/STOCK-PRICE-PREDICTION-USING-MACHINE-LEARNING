"""
Main script - Phase 1 Data Pipeline
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import StockDataFetcher
from src.feature_engineering import FeatureEngineer
from src.data_cleaner import DataCleaner
from src.database import DatabaseManager
from config.config import DATABASE_URL, DEFAULT_TICKERS
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def process_single_stock(ticker: str):
    """Process a single stock"""
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {ticker}")
    logger.info(f"{'='*60}\n")
    
    # Fetch data
    fetcher = StockDataFetcher()
    raw_data = fetcher.get_stock_data(ticker)
    
    if raw_data.empty:
        logger.error(f"No data for {ticker}")
        return None
    
    # Clean data
    cleaner = DataCleaner(raw_data)
    clean_data = cleaner.clean_all()
    
    # Add features
    engineer = FeatureEngineer(clean_data)
    processed_data = engineer.add_all_features()
    
    # Save to database
    db = DatabaseManager(DATABASE_URL)
    db.save_dataframe(processed_data)
    
    # Save to CSV
    csv_path = f'data/processed/{ticker}_processed.csv'
    processed_data.to_csv(csv_path, index=False)
    logger.info(f"Saved to {csv_path}")
    
    logger.info(f"\n✓ {ticker} complete! {len(processed_data)} rows, {len(processed_data.columns)} columns")
    
    return processed_data


def main():
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║      STOCK PREDICTION - PHASE 1: DATA PIPELINE          ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    
    print("\nSelect an option:")
    print("1. Process a single stock")
    print("2. Process multiple stocks (AAPL, GOOGL, MSFT, TSLA, AMZN)")
    print("3. Exit")
    
    choice = input("\nEnter choice (1-3): ")
    
    if choice == '1':
        ticker = input("Enter stock ticker: ").upper()
        process_single_stock(ticker)
    elif choice == '2':
        for ticker in DEFAULT_TICKERS:
            process_single_stock(ticker)
    else:
        print("Exiting...")
        return
    
    print("\n" + "="*60)
    print("PHASE 1 COMPLETE! ✓")
    print("="*60)


if __name__ == "__main__":
    main()