import matplotlib
matplotlib.use('Agg')
import numpy as np
import os
from eigenflow.utils import plot_regression_results, plot_loss_comparison

def test_plot_regression_results():
    """Verify that plot_regression_results generates and saves figures correctly."""
    losses = [10.0, 1.0, 0.1, 0.01]
    y_true = np.random.uniform(1.0, 10.0, size=(100,))
    y_pred = y_true + np.random.normal(0.0, 0.1, size=(100,))
    
    # 2D dummy inputs
    X_raw = np.random.uniform(-1.0, 1.0, size=(100, 2))
    y_raw = y_true
    
    save_path = "test_results.png"
    if os.path.exists(save_path):
        os.remove(save_path)
        
    try:
        # We specify save_path to render and save. In headless tests, saving to file is robust.
        plot_regression_results(
            losses=losses,
            y_true=y_true,
            y_pred=y_pred,
            test_losses=[(1, 1.0), (3, 0.01)],
            X_raw=X_raw,
            y_raw=y_raw,
            variables=["t", "v"],
            slice_feature_idx=1,
            title="Test Title",
            save_path=save_path,
            dpi=50 # low dpi to speed up testing
        )
        
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)

def test_plot_loss_comparison():
    """Verify that plot_loss_comparison generates and saves comparison plots correctly."""
    loss_dict = {
        "MLP": [10.0, 5.0, 2.0, 1.0],
        "KAN": [10.0, 2.0, 0.5, 0.1]
    }
    
    save_path = "test_comparison.png"
    if os.path.exists(save_path):
        os.remove(save_path)
        
    try:
        plot_loss_comparison(
            loss_dict=loss_dict,
            title="Test Loss Comparison",
            save_path=save_path,
            dpi=50
        )
        
        assert os.path.exists(save_path)
        assert os.path.getsize(save_path) > 0
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)
