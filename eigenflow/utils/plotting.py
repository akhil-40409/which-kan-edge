import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Optional, Union, Dict

def plot_regression_results(
    losses: Union[List[float], np.ndarray],
    y_true: Union[np.ndarray, List[float]],
    y_pred: Union[np.ndarray, List[float]],
    test_losses: Optional[List[Tuple[int, float]]] = None,
    X_raw: Optional[np.ndarray] = None,
    y_raw: Optional[np.ndarray] = None,
    variables: Optional[List[str]] = None,
    slice_feature_idx: Optional[int] = None,
    model_label: str = "Model Prediction",
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    dpi: int = 150
):
    """Generates a premium 3-panel visual report for symbolic regression evaluation.
    
    Panel 1: Log-scale training loss curve with optional test loss markers.
    Panel 2: Prediction vs Ground Truth scatter plot with y=x line.
    Panel 3: Physical curve relationship slice (locks other variables near median).
    
    Args:
        losses: List or array of training losses.
        y_true: Ground truth target values.
        y_pred: Model predicted target values.
        test_losses: List of tuples (epoch, loss_val) for test loss markers.
        X_raw: Raw physical inputs of shape (N, D).
        y_raw: Raw physical targets of shape (N,).
        variables: List of string names for variables.
        slice_feature_idx: Index of feature to plot on the x-axis in Panel 3.
        model_label: Legend label for the predictions.
        title: Overall title for the figure.
        save_path: File path to save the figure (e.g. 'results.png').
        dpi: High-quality rendering resolution.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Premium styling parameters
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=dpi)
    
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
    # --- Panel 1: Loss Curve ---
    axes[0].plot(losses, label="Train Loss", color="#1f77b4", lw=2, alpha=0.9)
    if test_losses:
        epochs_test, vals_test = zip(*test_losses)
        axes[0].scatter(epochs_test, vals_test, color="#d62728", label="Test Loss", zorder=3, s=40, edgecolors='k')
    axes[0].set_yscale("log")
    axes[0].set_title("Training Loss History", fontsize=13, fontweight='semibold')
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("MSE Loss", fontsize=11)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend(frameon=True, facecolor='white', framealpha=0.9)
    
    # --- Panel 2: Predictions vs Ground Truth ---
    axes[1].scatter(y_true, y_pred, alpha=0.4, color="#2ca02c", edgecolors='none', s=25)
    diag_min = min(y_true.min(), y_pred.min())
    diag_max = max(y_true.max(), y_pred.max())
    axes[1].plot([diag_min, diag_max], [diag_min, diag_max], "k--", lw=1.5, alpha=0.7)
    axes[1].set_title("Predictions vs Ground Truth", fontsize=13, fontweight='semibold')
    axes[1].set_xlabel("True Target Value", fontsize=11)
    axes[1].set_ylabel("Predicted Value", fontsize=11)
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    # --- Panel 3: Physical Relationship Slice ---
    if X_raw is not None and y_raw is not None:
        X_raw = np.array(X_raw)
        y_raw = np.array(y_raw)
        num_features = X_raw.shape[1]
        
        # Decide which feature to slice against
        if slice_feature_idx is None:
            slice_feature_idx = 1 if num_features > 1 else 0
            
        feat_name = variables[slice_feature_idx] if variables else f"Feature {slice_feature_idx}"
        
        # Slicing: keep other features near their medians
        mask = np.ones(len(X_raw), dtype=bool)
        slice_info = []
        
        for idx in range(num_features):
            if idx != slice_feature_idx:
                feat_vals = X_raw[:, idx]
                med = np.median(feat_vals)
                std = np.std(feat_vals)
                # Take values within 0.3 standard deviations of the median
                tol = 0.3 * std if std > 0 else 0.5
                mask = mask & (np.abs(feat_vals - med) <= tol)
                name = variables[idx] if variables else f"x{idx}"
                slice_info.append(f"{name} ≈ {med:.2f}")
                
        indices = np.where(mask)[0]
        # Fallback to all data if slice is too sparse
        if len(indices) < 10:
            indices = np.arange(len(X_raw))
            slice_info = ["Unconstrained Slice"]
            
        sort_idx = np.argsort(X_raw[indices, slice_feature_idx])
        sorted_x = X_raw[indices, slice_feature_idx][sort_idx]
        sorted_true = y_raw[indices][sort_idx]
        sorted_preds = y_pred[indices][sort_idx]
        
        axes[2].plot(sorted_x, sorted_true, label="Ground Truth (AI Feynman)", color="#ff7f0e", lw=2.5, alpha=0.9)
        axes[2].scatter(sorted_x, sorted_preds, label=model_label, color="#9467bd", alpha=0.6, s=25, edgecolors='none')
        
        axes[2].set_title(f"Target vs {feat_name}", fontsize=13, fontweight='semibold')
        axes[2].set_xlabel(feat_name, fontsize=11)
        axes[2].set_ylabel("Output Value", fontsize=11)
        axes[2].grid(True, linestyle="--", alpha=0.5)
        
        # Show slice metadata
        meta_text = ", ".join(slice_info)
        axes[2].text(0.05, 0.95, meta_text, transform=axes[2].transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='#cccccc'))
        axes[2].legend(frameon=True, facecolor='white', framealpha=0.9)
    else:
        # Fallback if raw inputs aren't provided
        axes[2].text(0.5, 0.5, "Physical Slice Plot\n(Requires X_raw and y_raw)", 
                     horizontalalignment='center', verticalalignment='center', fontsize=12, style='italic', color='#777777')
        axes[2].grid(False)
        
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Results plot saved to {save_path}")
    plt.show()

def plot_loss_comparison(
    loss_dict: Dict[str, Union[List[float], np.ndarray]],
    title: str = "Model Training Convergence Comparison",
    save_path: Optional[str] = None,
    dpi: int = 150
):
    """Plots training loss trajectories for multiple models on the same log-scale graph.
    
    Args:
        loss_dict: Dict mapping model name strings (e.g. 'MLP', 'KAN') to lists of loss history.
        title: Title of the comparison plot.
        save_path: File path to save the comparison plot.
        dpi: Quality resolution of figure.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(9, 6), dpi=dpi)
    
    # Custom color palette for models
    colors = ['#1f77b4', '#d62728', '#2ca02c', '#9467bd', '#ff7f0e']
    
    for i, (name, losses) in enumerate(loss_dict.items()):
        color = colors[i % len(colors)]
        plt.plot(losses, label=name, color=color, lw=2.5, alpha=0.85)
        
    plt.yscale("log")
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("MSE Loss", fontsize=12)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.legend(fontsize=11, frameon=True, facecolor='white', framealpha=0.9)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        print(f"Loss comparison plot saved to {save_path}")
    plt.show()
