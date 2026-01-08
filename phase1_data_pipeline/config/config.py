"""
Configuration settings for the data pipeline
"""
import os
from datetime import datetime, timedelta

# ========================
# DATA FETCHING CONFIG
# ========================
DEFAULT_START_DATE = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
DEFAULT_END_DATE = datetime.now().strftime('%Y-%m-%d')
DEFAULT_INTERVAL = '1d'

# Stock symbols to fetch
DEFAULT_TICKERS = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']

# ========================
# TECHNICAL INDICATORS CONFIG
# ========================
SMA_PERIODS = [20, 50, 200]
EMA_PERIODS = [12, 26]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
VOLATILITY_WINDOW = 30

# ========================
# DATA STORAGE CONFIG
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
DATABASE_PATH = os.path.join(DATA_DIR, 'stock_data.db')
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)

# ========================
# DATA VALIDATION CONFIG
# ========================
MIN_DATA_POINTS = 100
MAX_MISSING_PERCENTAGE = 5

# ========================
# LOGGING CONFIG
# ========================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'