"""
Main script for Phase 2 - Train and Compare Models
"""
import sys
import os
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import DataLoader
from src.feature_scaler import FeatureScaler
from src.sequence_creator import SequenceCreator
from src.baseline_models import BaselineModels
from src.lstm_model import LSTMModel
from src.model_evaluator import ModelEvaluator
from src.predictor import StockPredictor

from config.config import SAVED_MODELS_DIR

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_header(text):
    """Print a nice header"""
    print(f"\n{'='*80}")
    print(f"{text.center(80)}")
    print(f"{'='*80}\n")


def train_all_models(ticker: str):
    """
    Train all models for a given ticker
    
    Args:
        ticker: Stock symbol
    
    Returns:
        evaluator: ModelEvaluator with results
        best_model: Best performing model
        scaler: FeatureScaler object
    """
    print_header(f"🧠 TRAINING ML MODELS FOR {ticker}")
    
    # ================================
    # STEP 1: Load Data
    # ================================
    print_header("STEP 1: Loading Data")
    
    loader = DataLoader(ticker)
    data_split = loader.train_test_split()
    
    # ================================
    # STEP 2: Scale Features
    # ================================
    print_header("STEP 2: Scaling Features")
    
    scaler = FeatureScaler(method='minmax')
    scaled_data = scaler.fit_transform(
        data_split['X_train'],
        data_split['y_train'],
        data_split['X_test'],
        data_split['y_test']
    )
    
    # Save scaler
    scaler_path = os.path.join(SAVED_MODELS_DIR, f'{ticker}_scaler.pkl')
    scaler.save(scaler_path)
    
    # ================================
    # STEP 3: Create Model Evaluator
    # ================================
    evaluator = ModelEvaluator()
    
    # ================================
    # MODEL 1: Linear Regression
    # ================================
    print_header("MODEL 1: Linear Regression")
    
    start_time = time.time()
    
    lr_model = BaselineModels('linear_regression')
    lr_model.train(scaled_data['X_train_scaled'], scaled_data['y_train_scaled'])
    lr_metrics, lr_pred = lr_model.evaluate(
        scaled_data['X_test_scaled'],
        scaled_data['y_test_scaled']
    )
    
    lr_time = time.time() - start_time
    
    # Inverse transform predictions
    lr_pred_actual = scaler.inverse_transform_y(lr_pred)
    
    evaluator.add_model_results(
        'Linear Regression',
        data_split['y_test'],
        lr_pred_actual,
        training_time=lr_time
    )
    
    # Save model
    lr_path = os.path.join(SAVED_MODELS_DIR, f'{ticker}_linear_regression.pkl')
    lr_model.save(lr_path)
    
    # ================================
    # MODEL 2: Random Forest
    # ================================
    print_header("MODEL 2: Random Forest")
    
    start_time = time.time()
    
    rf_model = BaselineModels('random_forest')
    rf_model.train(scaled_data['X_train_scaled'], scaled_data['y_train_scaled'])
    rf_metrics, rf_pred = rf_model.evaluate(
        scaled_data['X_test_scaled'],
        scaled_data['y_test_scaled']
    )
    
    rf_time = time.time() - start_time
    
    rf_pred_actual = scaler.inverse_transform_y(rf_pred)
    
    evaluator.add_model_results(
        'Random Forest',
        data_split['y_test'],
        rf_pred_actual,
        training_time=rf_time
    )
    
    # Save model
    rf_path = os.path.join(SAVED_MODELS_DIR, f'{ticker}_random_forest.pkl')
    rf_model.save(rf_path)
    
    # ================================
    # MODEL 3: LSTM
    # ================================
    print_header("MODEL 3: LSTM Neural Network")
    
    # Create sequences
    seq_creator = SequenceCreator(sequence_length=60)
    X_train_seq, y_train_seq = seq_creator.create_sequences(
        scaled_data['X_train_scaled'],
        scaled_data['y_train_scaled']
    )
    X_test_seq, y_test_seq = seq_creator.create_sequences(
        scaled_data['X_test_scaled'],
        scaled_data['y_test_scaled']
    )
    
    start_time = time.time()
    
    # Build LSTM
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    lstm = LSTMModel(input_shape=input_shape, lstm_units=[100, 50], dropout_rate=0.2)
    
    # Train LSTM
    history = lstm.train(
        X_train_seq, y_train_seq,
        X_val=X_test_seq, y_val=y_test_seq,
        epochs=50,
        batch_size=32,
        verbose=1
    )
    
    lstm_time = time.time() - start_time
    
    # Evaluate LSTM
    lstm_metrics, lstm_pred = lstm.evaluate(X_test_seq, y_test_seq)
    
    # Inverse transform
    lstm_pred_actual = scaler.inverse_transform_y(lstm_pred)
    y_test_actual = scaler.inverse_transform_y(y_test_seq)
    
    evaluator.add_model_results(
        'LSTM',
        y_test_actual,
        lstm_pred_actual,
        training_time=lstm_time
    )
    
    # Save LSTM
    lstm_path = os.path.join(SAVED_MODELS_DIR, f'{ticker}_lstm.h5')
    lstm.save(lstm_path)
    
    # ================================
    # STEP 4: Compare Models
    # ================================
    print_header("MODEL COMPARISON")
    
    comparison_df = evaluator.print_comparison()
    
    # Plot comparisons
    evaluator.plot_predictions(dates=data_split['dates_test'][-len(lr_pred_actual):])
    evaluator.plot_metrics_comparison()
    
    # Save results
    results_path = os.path.join(SAVED_MODELS_DIR, f'{ticker}_model_results.json')
    evaluator.save_results(results_path)
    
    # ================================
    # Determine Best Model
    # ================================
    best_model_name = comparison_df.iloc[0]['Model']
    
    if best_model_name == 'Linear Regression':
        best_model = lr_model
        seq_creator_for_pred = None
    elif best_model_name == 'Random Forest':
        best_model = rf_model
        seq_creator_for_pred = None
    else:  # LSTM
        best_model = lstm
        seq_creator_for_pred = seq_creator
    
    logger.info(f"\n🏆 Best Model: {best_model_name}\n")
    
    return evaluator, best_model, scaler, seq_creator_for_pred, loader


def make_future_predictions(ticker, model, scaler, seq_creator, loader, n_days=30):
    """
    Make future predictions
    
    Args:
        ticker: Stock symbol
        model: Trained model
        scaler: FeatureScaler
        seq_creator: SequenceCreator (or None)
        loader: DataLoader
        n_days: Number of days to predict
    """
    print_header(f"🔮 PREDICTING NEXT {n_days} DAYS FOR {ticker}")
    
    # Get latest data
    latest_data = loader.get_latest_data(n_days=60)
    
    # Create predictor
    predictor = StockPredictor(model, scaler, seq_creator)
    
    # Make predictions
    predictions, dates = predictor.predict_with_confidence(latest_data, n_days=n_days)
    
    # Create DataFrame
    pred_df = predictor.create_prediction_df(predictions, dates, ticker)
    
    # Save predictions
    pred_path = os.path.join('data', 'predictions', f'{ticker}_predictions.csv')
    pred_df.to_csv(pred_path, index=False)
    
    logger.info(f"✓ Predictions saved to {pred_path}\n")
    
    # Display predictions
    print("\nPredicted Prices (Next 10 Days):")
    print("="*60)
    print(f"{'Date':<12} {'Price':<12} {'Lower':<12} {'Upper':<12}")
    print("="*60)
    
    for i in range(min(10, len(pred_df))):
        row = pred_df.iloc[i]
        print(f"{row['date'].strftime('%Y-%m-%d'):<12} "
              f"${row['predicted_price']:>8.2f}    "
              f"${row['lower_bound']:>8.2f}    "
              f"${row['upper_bound']:>8.2f}")
    
    print("="*60 + "\n")
    
    return pred_df


def main():
    """Main execution"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║           STOCK PREDICTION - PHASE 2: MACHINE LEARNING ENGINE               ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print("\nSelect an option:")
    print("1. Train models for a single stock")
    print("2. Train models for multiple stocks (AAPL, GOOGL, MSFT, TSLA, AMZN)")
    print("3. Make predictions with existing model")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
        
        # Train models
        evaluator, best_model, scaler, seq_creator, loader = train_all_models(ticker)
        
        # Make predictions
        make_pred = input("\nMake future predictions? (y/n): ").lower()
        if make_pred == 'y':
            n_days = int(input("How many days to predict? (default: 30): ") or "30")
            make_future_predictions(ticker, best_model, scaler, seq_creator, loader, n_days)
    
    elif choice == '2':
        tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        
        for ticker in tickers:
            try:
                evaluator, best_model, scaler, seq_creator, loader = train_all_models(ticker)
                make_future_predictions(ticker, best_model, scaler, seq_creator, loader, n_days=30)
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue
    
    elif choice == '3':
        ticker = input("Enter stock ticker: ").upper()
        # Load existing model (implementation left as exercise)
        logger.info("Feature coming soon!")
    
    else:
        logger.info("Exiting...")
        return
    
    print("\n" + "="*80)
    print("PHASE 2 COMPLETE! ✓")
    print("="*80)
    print("\nYour models are trained and ready!")
    print("\nNext steps:")
    print("1. Check 'models/saved_models/' for trained models")
    print("2. Check 'data/predictions/' for future predictions")
    print("3. Review the comparison plots generated")
    print("4. Proceed to Phase 3 for Web Dashboard (Coming next!)")
    print()


if __name__ == "__main__":
    main()