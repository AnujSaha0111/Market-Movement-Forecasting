"""Model interpretability using SHAP."""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional

# SHAP is optional - may not be available on Windows without build tools
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("SHAP not available. Model interpretability features will be limited.")

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class ModelInterpreter:
    """Model interpretability using SHAP values."""
    
    def __init__(self):
        """Initialize interpreter."""
        self.explainers = {}
        self.shap_values = {}
    
    def explain_model(self, model, X: np.ndarray, model_name: str = 'Model',
                     model_type: str = 'tree', max_samples: int = 100) -> Optional[object]:
        """
        Create SHAP explainer for a model.
        
        Args:
            model: Trained model
            X: Training/background data
            model_name: Name of the model
            model_type: Type of model ('tree', 'linear', 'deep', 'kernel')
            max_samples: Maximum samples for background data
            
        Returns:
            SHAP explainer or None if SHAP not available
        """
        if not SHAP_AVAILABLE:
            logger.warning(f"SHAP not available. Cannot create explainer for {model_name}")
            return None
        
        logger.info(f"Creating SHAP explainer for {model_name} ({model_type})")
        
        # Limit background data size
        if len(X) > max_samples:
            background_data = shap.sample(X, max_samples)
        else:
            background_data = X
        
        # Create explainer based on model type
        if model_type == 'tree':
            explainer = shap.TreeExplainer(model)
        elif model_type == 'linear':
            explainer = shap.LinearExplainer(model, background_data)
        elif model_type == 'deep':
            explainer = shap.DeepExplainer(model, background_data)
        elif model_type == 'kernel':
            explainer = shap.KernelExplainer(model.predict_proba, background_data)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.explainers[model_name] = explainer
        logger.info(f"Created {model_type} explainer for {model_name}")
        
        return explainer
    
    def calculate_shap_values(self, model_name: str, X: np.ndarray,
                             max_samples: int = None) -> np.ndarray:
        """
        Calculate SHAP values.
        
        Args:
            model_name: Name of the model
            X: Data to explain
            max_samples: Maximum samples to explain
            
        Returns:
            SHAP values
        """
        if model_name not in self.explainers:
            raise ValueError(f"No explainer found for {model_name}. Create one first.")
        
        logger.info(f"Calculating SHAP values for {model_name}")
        
        # Limit samples if needed
        if max_samples and len(X) > max_samples:
            X_sample = shap.sample(X, max_samples)
        else:
            X_sample = X
        
        explainer = self.explainers[model_name]
        shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification, take positive class
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        
        self.shap_values[model_name] = shap_values
        logger.info(f"Calculated SHAP values shape: {shap_values.shape}")
        
        return shap_values
    
    def get_feature_importance(self, model_name: str, feature_names: List[str],
                              top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from SHAP values.
        
        Args:
            model_name: Name of the model
            feature_names: List of feature names
            top_n: Number of top features
            
        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_name}. Calculate them first.")
        
        shap_values = self.shap_values[model_name]
        
        # Calculate mean absolute SHAP values
        importance = np.abs(shap_values).mean(axis=0)
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        logger.info(f"\nTop {top_n} features for {model_name}:")
        logger.info(f"\n{importance_df.to_string()}")
        
        return importance_df
    
    def plot_summary(self, model_name: str, X: np.ndarray, feature_names: List[str],
                    save_path: Optional[Path] = None, max_display: int = 20) -> None:
        """
        Create SHAP summary plot.
        
        Args:
            model_name: Name of the model
            X: Data used for SHAP values
            feature_names: List of feature names
            save_path: Path to save plot
            max_display: Maximum features to display
        """
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_name}")
        
        logger.info(f"Creating SHAP summary plot for {model_name}")
        
        shap_values = self.shap_values[model_name]
        
        # Ensure X matches shap_values shape
        if len(X) != len(shap_values):
            X = X[:len(shap_values)]
        
        # Create plot
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X,
            feature_names=feature_names,
            max_display=max_display,
            show=False
        )
        
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            plot_file = save_path / f"{model_name}_shap_summary.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP summary plot to {plot_file}")
        
        plt.close()
    
    def plot_waterfall(self, model_name: str, X: np.ndarray, feature_names: List[str],
                      instance_idx: int = 0, save_path: Optional[Path] = None) -> None:
        """
        Create SHAP waterfall plot for a single prediction.
        
        Args:
            model_name: Name of the model
            X: Data used for SHAP values
            feature_names: List of feature names
            instance_idx: Index of instance to explain
            save_path: Path to save plot
        """
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_name}")
        
        logger.info(f"Creating SHAP waterfall plot for {model_name}, instance {instance_idx}")
        
        shap_values = self.shap_values[model_name]
        explainer = self.explainers[model_name]
        
        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values[instance_idx],
            base_values=explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
            data=X[instance_idx],
            feature_names=feature_names
        )
        
        # Create plot
        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(explanation, show=False)
        
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            plot_file = save_path / f"{model_name}_shap_waterfall_{instance_idx}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP waterfall plot to {plot_file}")
        
        plt.close()
    
    def plot_force(self, model_name: str, X: np.ndarray, feature_names: List[str],
                  instance_idx: int = 0, save_path: Optional[Path] = None) -> None:
        """
        Create SHAP force plot for a single prediction.
        
        Args:
            model_name: Name of the model
            X: Data used for SHAP values
            feature_names: List of feature names
            instance_idx: Index of instance to explain
            save_path: Path to save plot
        """
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_name}")
        
        logger.info(f"Creating SHAP force plot for {model_name}, instance {instance_idx}")
        
        shap_values = self.shap_values[model_name]
        explainer = self.explainers[model_name]
        
        # Create force plot
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value if hasattr(explainer, 'expected_value') else 0,
            shap_values[instance_idx],
            X[instance_idx],
            feature_names=feature_names
        )
        
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            plot_file = save_path / f"{model_name}_shap_force_{instance_idx}.html"
            shap.save_html(str(plot_file), force_plot)
            logger.info(f"Saved SHAP force plot to {plot_file}")
    
    def plot_dependence(self, model_name: str, X: np.ndarray, feature_names: List[str],
                       feature: str, save_path: Optional[Path] = None) -> None:
        """
        Create SHAP dependence plot for a feature.
        
        Args:
            model_name: Name of the model
            X: Data used for SHAP values
            feature_names: List of feature names
            feature: Feature to plot
            save_path: Path to save plot
        """
        if model_name not in self.shap_values:
            raise ValueError(f"No SHAP values found for {model_name}")
        
        logger.info(f"Creating SHAP dependence plot for {model_name}, feature {feature}")
        
        shap_values = self.shap_values[model_name]
        
        # Get feature index
        if feature in feature_names:
            feature_idx = feature_names.index(feature)
        else:
            raise ValueError(f"Feature {feature} not found in feature names")
        
        # Create plot
        plt.figure(figsize=(10, 6))
        shap.dependence_plot(
            feature_idx, shap_values, X,
            feature_names=feature_names,
            show=False
        )
        
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            plot_file = save_path / f"{model_name}_shap_dependence_{feature}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Saved SHAP dependence plot to {plot_file}")
        
        plt.close()


if __name__ == "__main__":
    # Test interpreter
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    
    # Generate data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Train model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Interpret
    interpreter = ModelInterpreter()
    interpreter.explain_model(model, X_train, 'RandomForest', model_type='tree')
    interpreter.calculate_shap_values('RandomForest', X_test, max_samples=100)
    
    # Get feature importance
    importance = interpreter.get_feature_importance('RandomForest', feature_names, top_n=10)
    print("\nFeature Importance:")
    print(importance)
