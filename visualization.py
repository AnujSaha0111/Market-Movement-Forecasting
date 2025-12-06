"""Visualization utilities for model results."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


class Visualizer:
    """Visualization utilities for model evaluation."""
    
    def __init__(self, save_path: Optional[Path] = None):
        """
        Initialize visualizer.
        
        Args:
            save_path: Default path to save plots
        """
        self.save_path = save_path
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
    
    def plot_confusion_matrix(self, cm: np.ndarray, model_name: str = 'Model',
                             save_name: Optional[str] = None) -> None:
        """Plot confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_name and self.save_path:
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            logger.info(f"Saved confusion matrix to {self.save_path / save_name}")
        
        plt.close()
    
    def plot_roc_curve(self, results_list: List[Dict], save_name: Optional[str] = None) -> None:
        """Plot ROC curves for multiple models."""
        plt.figure(figsize=(10, 8))
        
        for result in results_list:
            if 'roc_curve' in result:
                fpr = result['roc_curve']['fpr']
                tpr = result['roc_curve']['tpr']
                auc = result.get('roc_auc', 0)
                model_name = result.get('model', 'Unknown')
                
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=14)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_name and self.save_path:
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            logger.info(f"Saved ROC curve to {self.save_path / save_name}")
        
        plt.close()
    
    def plot_precision_recall_curve(self, results_list: List[Dict],
                                    save_name: Optional[str] = None) -> None:
        """Plot precision-recall curves for multiple models."""
        plt.figure(figsize=(10, 8))
        
        for result in results_list:
            if 'pr_curve' in result:
                precision = result['pr_curve']['precision']
                recall = result['pr_curve']['recall']
                model_name = result.get('model', 'Unknown')
                
                plt.plot(recall, precision, label=model_name, linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves', fontsize=14)
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_name and self.save_path:
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            logger.info(f"Saved PR curve to {self.save_path / save_name}")
        
        plt.close()
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, model_name: str = 'Model',
                               save_name: Optional[str] = None, top_n: int = 20) -> None:
        """Plot feature importance."""
        plt.figure(figsize=(10, 8))
        
        # Get top features
        plot_df = importance_df.head(top_n).sort_values('importance')
        
        plt.barh(range(len(plot_df)), plot_df['importance'])
        plt.yticks(range(len(plot_df)), plot_df['feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.title(f'Top {top_n} Feature Importance - {model_name}', fontsize=14)
        plt.tight_layout()
        
        if save_name and self.save_path:
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            logger.info(f"Saved feature importance to {self.save_path / save_name}")
        
        plt.close()
    
    def plot_model_comparison(self, comparison_df: pd.DataFrame,
                             save_name: Optional[str] = None) -> None:
        """Plot model comparison."""
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        
        fig, axes = plt.subplots(1, len(metrics), figsize=(20, 4))
        
        for idx, metric in enumerate(metrics):
            if metric in comparison_df.columns:
                axes[idx].bar(comparison_df['Model'], comparison_df[metric])
                axes[idx].set_title(metric, fontsize=12)
                axes[idx].set_ylabel('Score', fontsize=10)
                axes[idx].tick_params(axis='x', rotation=45)
                axes[idx].set_ylim([0, 1])
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name and self.save_path:
            plt.savefig(self.save_path / save_name, dpi=300, bbox_inches='tight')
            logger.info(f"Saved model comparison to {self.save_path / save_name}")
        
        plt.close()
    
    def plot_cumulative_returns(self, df: pd.DataFrame, model_name: str = 'Model',
                               save_name: Optional[str] = None) -> None:
        """Plot cumulative returns comparison."""
        fig = go.Figure()
        
        # Buy and hold
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['cumulative_return'] * 100,
            mode='lines',
            name='Buy & Hold',
            line=dict(color='blue', width=2)
        ))
        
        # Strategy
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['strategy_cumulative_return'] * 100,
            mode='lines',
            name=f'{model_name} Strategy',
            line=dict(color='green', width=2)
        ))
        
        fig.update_layout(
            title=f'Cumulative Returns - {model_name}',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)',
            hovermode='x unified',
            template='plotly_white'
        )
        
        if save_name and self.save_path:
            fig.write_html(str(self.save_path / save_name))
            logger.info(f"Saved cumulative returns to {self.save_path / save_name}")
    
    def plot_price_and_predictions(self, df: pd.DataFrame, symbol: str = 'Stock',
                                   save_name: Optional[str] = None) -> None:
        """Plot stock price with predictions."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Stock Price', 'Predictions')
        )
        
        # Price
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(x=df.index, y=df['prediction'], name='Prediction', 
                      mode='markers', marker=dict(color='green', size=4)),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'{symbol} - Price and Predictions',
            hovermode='x unified',
            template='plotly_white',
            height=800
        )
        
        if save_name and self.save_path:
            fig.write_html(str(self.save_path / save_name))
            logger.info(f"Saved price and predictions to {self.save_path / save_name}")
    
    def plot_sentiment_timeline(self, df: pd.DataFrame, symbol: str = 'Stock',
                               save_name: Optional[str] = None) -> None:
        """Plot sentiment over time."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=('Stock Price', 'Sentiment')
        )
        
        # Price
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], name='Close Price', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Sentiment
        if 'sentiment_combined' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['sentiment_combined'], name='Sentiment',
                          line=dict(color='purple')),
                row=2, col=1
            )
        
        fig.update_layout(
            title=f'{symbol} - Price and Sentiment',
            hovermode='x unified',
            template='plotly_white',
            height=800
        )
        
        if save_name and self.save_path:
            fig.write_html(str(self.save_path / save_name))
            logger.info(f"Saved sentiment timeline to {self.save_path / save_name}")
    
    def create_dashboard(self, results: Dict, comparison_df: pd.DataFrame,
                        save_name: str = 'dashboard.html') -> None:
        """Create comprehensive dashboard."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Comparison - Accuracy', 'Model Comparison - ROC AUC',
                          'Confusion Matrix', 'Feature Importance'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'heatmap'}, {'type': 'bar'}]]
        )
        
        # Accuracy comparison
        fig.add_trace(
            go.Bar(x=comparison_df['Model'], y=comparison_df['Accuracy'], name='Accuracy'),
            row=1, col=1
        )
        
        # AUC comparison
        fig.add_trace(
            go.Bar(x=comparison_df['Model'], y=comparison_df['ROC AUC'], name='ROC AUC'),
            row=1, col=2
        )
        
        fig.update_layout(
            title='Model Evaluation Dashboard',
            showlegend=False,
            template='plotly_white',
            height=800
        )
        
        if self.save_path:
            fig.write_html(str(self.save_path / save_name))
            logger.info(f"Saved dashboard to {self.save_path / save_name}")


if __name__ == "__main__":
    # Test visualizer
    import tempfile
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = Visualizer(save_path=Path(tmpdir))
        
        # Test confusion matrix
        cm = np.array([[50, 10], [5, 35]])
        viz.plot_confusion_matrix(cm, 'Test Model', 'test_cm.png')
        
        print("Visualizer test complete")
