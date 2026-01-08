"""
Model evaluation and comparison utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluate and compare multiple models
    """
    
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name: str, y_true, y_pred, training_time=None):
        """
        Add model predictions for evaluation
        
        Args:
            model_name: Name of the model
            y_true: True values
            y_pred: Predicted values
            training_time: Training time in seconds (optional)
        """
        metrics = self.calculate_metrics(y_true, y_pred)
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'training_time': training_time
        }
        
        logger.info(f"✓ Added results for {model_name}")
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate all evaluation metrics"""
        
        # Ensure arrays
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        
        # Basic metrics
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # MAPE (avoid division by zero)
        mask = y_true != 0
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        
        # Directional Accuracy
        if len(y_true) > 1:
            actual_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        else:
            directional_accuracy = 0
        
        # Additional metrics
        max_error = np.max(np.abs(y_true - y_pred))
        
        metrics = {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Directional_Accuracy': directional_accuracy,
            'Max_Error': max_error
        }
        
        return metrics
    
    def print_comparison(self):
        """Print comparison table of all models"""
        
        logger.info(f"\n{'='*80}")
        logger.info("MODEL COMPARISON")
        logger.info(f"{'='*80}\n")
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, data in self.results.items():
            row = {'Model': model_name}
            row.update(data['metrics'])
            if data['training_time']:
                row['Training_Time'] = f"{data['training_time']:.2f}s"
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        df = df.sort_values('RMSE')
        
        print(df.to_string(index=False))
        
        logger.info(f"\n{'='*80}\n")
        
        # Determine best model
        best_model = df.iloc[0]['Model']
        logger.info(f"🏆 BEST MODEL: {best_model}")
        logger.info(f"   RMSE: {df.iloc[0]['RMSE']:.4f}")
        logger.info(f"   MAE: {df.iloc[0]['MAE']:.4f}")
        logger.info(f"   R²: {df.iloc[0]['R2']:.4f}")
        logger.info(f"   Directional Accuracy: {df.iloc[0]['Directional_Accuracy']:.2f}%\n")
        
        return df
    
    def plot_predictions(self, dates=None, figsize=(15, 10)):
        """
        Plot predictions vs actual for all models
        
        Args:
            dates: Array of dates for x-axis (optional)
            figsize: Figure size
        """
        n_models = len(self.results)
        
        if n_models == 0:
            logger.warning("No models to plot")
            return
        
        # Create subplots
        fig, axes = plt.subplots(n_models, 1, figsize=figsize)
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, data) in enumerate(self.results.items()):
            ax = axes[idx]
            
            y_true = data['y_true']
            y_pred = data['y_pred']
            
            if dates is not None and len(dates) == len(y_true):
                x = dates
            else:
                x = np.arange(len(y_true))
            
            # Plot
            ax.plot(x, y_true, label='Actual', color='blue', linewidth=2, alpha=0.7)
            ax.plot(x, y_pred, label='Predicted', color='red', linewidth=2, alpha=0.7)
            
            # Styling
            ax.set_title(f'{model_name} - RMSE: {data["metrics"]["RMSE"]:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time')
            ax.set_ylabel('Price')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_predictions_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved predictions plot to 'model_predictions_comparison.png'")
        plt.show()
    
    def plot_metrics_comparison(self, figsize=(12, 6)):
        """Plot bar chart comparing metrics across models"""
        
        if len(self.results) == 0:
            logger.warning("No models to plot")
            return
        
        # Prepare data
        models = list(self.results.keys())
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'Directional_Accuracy']
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            
            values = [self.results[model]['metrics'][metric] for model in models]
            
            bars = ax.bar(models, values, alpha=0.7, color='steelblue')
            
            # Highlight best
            if metric in ['R2', 'Directional_Accuracy']:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            
            bars[best_idx].set_color('green')
            
            ax.set_title(metric, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x labels if needed
            if len(models) > 3:
                ax.set_xticklabels(models, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        logger.info("✓ Saved metrics comparison to 'metrics_comparison.png'")
        plt.show()
    
    def save_results(self, filepath: str):
        """Save evaluation results to JSON"""
        
        # Prepare data for JSON
        results_json = {}
        
        for model_name, data in self.results.items():
            results_json[model_name] = {
                'metrics': data['metrics'],
                'training_time': data['training_time']
            }
        
        # Add timestamp
        results_json['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        with open(filepath, 'w') as f:
            json.dump(results_json, f, indent=4)
        
        logger.info(f"✓ Results saved to {filepath}")


if __name__ == "__main__":
    # Test evaluator
    np.random.seed(42)
    
    y_true = np.sin(np.linspace(0, 10, 100)) + np.random.normal(0, 0.1, 100)
    
    # Simulate different model predictions
    y_pred_lr = y_true + np.random.normal(0, 0.2, 100)
    y_pred_rf = y_true + np.random.normal(0, 0.15, 100)
    y_pred_lstm = y_true + np.random.normal(0, 0.1, 100)
    
    # Create evaluator
    evaluator = ModelEvaluator()
    
    # Add model results
    evaluator.add_model_results('Linear Regression', y_true, y_pred_lr, training_time=0.5)
    evaluator.add_model_results('Random Forest', y_true, y_pred_rf, training_time=2.3)
    evaluator.add_model_results('LSTM', y_true, y_pred_lstm, training_time=45.7)
    
    # Print comparison
    df = evaluator.print_comparison()
    
    # Plot results
    evaluator.plot_predictions()
    evaluator.plot_metrics_comparison()