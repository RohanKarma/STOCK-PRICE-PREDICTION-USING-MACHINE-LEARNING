"""
Create sequences for LSTM time series prediction
"""
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SequenceCreator:
    """
    Create sequences for LSTM model
    """
    
    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
    
    def create_sequences(self, X, y):
        """
        Create sequences for LSTM
        
        Args:
            X: Features array (n_samples, n_features)
            y: Target array (n_samples,)
        
        Returns:
            X_seq: (n_sequences, sequence_length, n_features)
            y_seq: (n_sequences,)
        """
        X_seq = []
        y_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        logger.info(f"Created sequences: {X_seq.shape}")
        
        return X_seq, y_seq
    
    def create_future_sequence(self, X_latest):
        """
        Create a single sequence from the latest data for prediction
        
        Args:
            X_latest: Latest n days of data (sequence_length, n_features)
        
        Returns:
            Sequence ready for prediction (1, sequence_length, n_features)
        """
        if len(X_latest) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} days of data")
        
        # Take the last sequence_length days
        sequence = X_latest[-self.sequence_length:]
        
        # Reshape for prediction
        sequence = sequence.reshape(1, self.sequence_length, -1)
        
        return sequence


if __name__ == "__main__":
    # Test sequence creator
    from data_loader import DataLoader
    from feature_scaler import FeatureScaler
    
    loader = DataLoader('AAPL')
    data_split = loader.train_test_split()
    
    scaler = FeatureScaler()
    scaled_data = scaler.fit_transform(
        data_split['X_train'],
        data_split['y_train'],
        data_split['X_test'],
        data_split['y_test']
    )
    
    seq_creator = SequenceCreator(sequence_length=60)
    X_train_seq, y_train_seq = seq_creator.create_sequences(
        scaled_data['X_train_scaled'],
        scaled_data['y_train_scaled']
    )
    
    print(f"\nOriginal training data: {scaled_data['X_train_scaled'].shape}")
    print(f"Sequenced training data: {X_train_seq.shape}")
    print(f"Training targets: {y_train_seq.shape}")