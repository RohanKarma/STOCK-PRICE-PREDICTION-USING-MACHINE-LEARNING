"""
Configuration for Phase 2 - Machine Learning
"""
import os
from datetime import datetime

# ========================
# PATHS
# ========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
PREDICTIONS_DIR = os.path.join(DATA_DIR, 'predictions')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
SAVED_MODELS_DIR = os.path.join(MODELS_DIR, 'saved_models')

# Link to Phase 1 data
PHASE1_DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), 'phase1_data_pipeline', 'data', 'processed')

# Create directories
for directory in [DATA_DIR, PROCESSED_DATA_DIR, PREDICTIONS_DIR, MODELS_DIR, SAVED_MODELS_DIR]:
    os.makedirs(directory, exist_ok=True)

# ========================
# DATA PREPARATION
# ========================
# Features to use for prediction
FEATURE_COLUMNS = [
    'open', 'high', 'low', 'close', 'volume',
    'sma_20', 'sma_50', 'sma_200',
    'ema_20', 'ema_50', 'ema_200',
    'rsi', 'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'atr', 'volatility',
    'volume_sma_20', 'volume_ratio',
    'obv', 'roc', 'stoch_k', 'stoch_d'
]

# Target variable
TARGET_COLUMN = 'close'

# Train/Test split
TRAIN_SIZE = 0.8  # 80% training, 20% testing

# ========================
# LSTM CONFIGURATION
# ========================
SEQUENCE_LENGTH = 60  # Use 60 days to predict the next day
LSTM_UNITS = [100, 50]  # Two LSTM layers
DROPOUT_RATE = 0.2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# ========================
# RANDOM FOREST CONFIGURATION
# ========================
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
RF_MIN_SAMPLES_SPLIT = 5

# ========================
# MODEL EVALUATION
# ========================
METRICS = ['RMSE', 'MAE', 'MAPE', 'R2', 'Directional_Accuracy']

# ========================
# PREDICTION
# ========================
PREDICTION_DAYS = 30  # Predict next 30 days
CONFIDENCE_INTERVAL = 0.95

# ========================
# LOGGING
# ========================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'