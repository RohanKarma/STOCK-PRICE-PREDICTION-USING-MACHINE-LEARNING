"""
LSTM Neural Network for Stock Price Prediction
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMModel:
    """
    LSTM Neural Network for time series prediction
    """
    
    def __init__(self, input_shape, lstm_units=[100, 50], dropout_rate=0.2):
        """
        Args:
            input_shape: (sequence_length, n_features)
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.history = None
        
        self._build_model()
    
    def _build_model(self):
        """Build the LSTM architecture"""
        logger.info("Building LSTM model...")
        
        self.model = Sequential()
        
        # First LSTM layer
        self.model.add(LSTM(
            units=self.lstm_units[0],
            return_sequences=True if len(self.lstm_units) > 1 else False,
            input_shape=self.input_shape
        ))
        self.model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i, units in enumerate(self.lstm_units[1:]):
            return_seq = i < len(self.lstm_units) - 2
            self.model.add(LSTM(units=units, return_sequences=return_seq))
            self.model.add(Dropout(self.dropout_rate))
        
        # Output layer
        self.model.add(Dense(units=1))
        
        # Compile model
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        logger.info(f"✓ LSTM model built successfully")
        self._print_model_summary()
    
    def _print_model_summary(self):
        """Print model architecture"""
        logger.info("\n" + "="*60)
        logger.info("LSTM MODEL ARCHITECTURE")
        logger.info("="*60)
        
        # Capture summary
        summary_lines = []
        self.model.summary(print_fn=lambda x: summary_lines.append(x))
        for line in summary_lines:
            logger.info(line)
        
        logger.info("="*60 + "\n")
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, verbose=1):
        """
        Train the LSTM model
        
        Args:
            X_train: Training sequences (n_samples, sequence_length, n_features)
            y_train: Training targets (n_samples,)
            X_val: Validation sequences (optional)
            y_val: Validation targets (optional)
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity level
        """
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING LSTM MODEL")
        logger.info(f"{'='*60}")
        logger.info(f"Training samples: {len(X_train)}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Batch size: {batch_size}")
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Prepare validation data
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
            logger.info(f"Validation samples: {len(X_val)}")
        
        logger.info(f"{'='*60}\n")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )
        
        logger.info(f"\n✓ Training complete!")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input sequences (n_samples, sequence_length, n_features)
        
        Returns:
            predictions: Array of predictions
        """
        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test: Test sequences
            y_test: True values
        
        Returns:
            metrics: Dictionary of evaluation metrics
            predictions: Model predictions
        """
        logger.info(f"\n{'='*60}")
        logger.info("EVALUATING LSTM MODEL")
        logger.info(f"{'='*60}")
        
        # Make predictions
        predictions = self.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        # MAPE
        mape = np.mean(np.abs((y_test - predictions) / y_test)) * 100
        
        # Directional Accuracy
        if len(y_test) > 1:
            actual_direction = np.diff(y_test) > 0
            pred_direction = np.diff(predictions) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy
        }
        
        # Print metrics
        for metric, value in metrics.items():
            logger.info(f"{metric:<25}: {value:.4f}")
        
        logger.info(f"{'='*60}\n")
        
        return metrics, predictions
    
    def save(self, filepath: str):
        """Save model to disk"""
        self.model.save(filepath)
        logger.info(f"✓ Model saved to {filepath}")
    
    def load(self, filepath: str):
        """Load model from disk"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"✓ Model loaded from {filepath}")
    
    def predict_future(self, last_sequence, n_steps=30):
        """
        Predict multiple steps into the future
        
        Args:
            last_sequence: Latest sequence (1, sequence_length, n_features)
            n_steps: Number of steps to predict
        
        Returns:
            predictions: Array of future predictions
        """
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(n_steps):
            # Predict next value
            next_pred = self.model.predict(current_sequence, verbose=0)[0, 0]
            predictions.append(next_pred)
            
            # Update sequence (shift left and add prediction)
            # For simplicity, we'll just use the prediction for the 'close' feature
            # In practice, you'd need to predict all features or use a different approach
            new_row = current_sequence[0, -1, :].copy()
            new_row[4] = next_pred  # Assuming 'close' is at index 4
            
            current_sequence = np.append(current_sequence[0, 1:, :], [new_row], axis=0)
            current_sequence = current_sequence.reshape(1, -1, current_sequence.shape[1])
        
        return np.array(predictions)


if __name__ == "__main__":
    # Test LSTM model
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    
    from data_loader import DataLoader
    from feature_scaler import FeatureScaler
    from sequence_creator import SequenceCreator
    
    print("\n" + "="*60)
    print("TESTING LSTM MODEL")
    print("="*60 + "\n")
    
    # Load and prepare data
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
    
    # Build and train LSTM
    input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
    lstm = LSTMModel(input_shape=input_shape, lstm_units=[100, 50], dropout_rate=0.2)
    
    # Train
    history = lstm.train(
        X_train_seq, y_train_seq,
        X_val=X_test_seq, y_val=y_test_seq,
        epochs=20,  # Reduced for testing
        batch_size=32
    )
    
    # Evaluate
    metrics, predictions = lstm.evaluate(X_test_seq, y_test_seq)