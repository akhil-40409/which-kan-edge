import os
import tempfile

import numpy as np

from eigenflow.utils.plotting import plot_loss_comparison, plot_regression_results


def test_plot_regression_results():
    losses = list(np.linspace(1.0, 0.1, 20))
    y_true = np.linspace(-1, 1, 50)
    y_pred = y_true + 0.05 * np.random.randn(50)
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "reg.png")
        plot_regression_results(losses, y_true, y_pred, save_path=path)
        assert os.path.isfile(path)


def test_plot_loss_comparison():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cmp.png")
        plot_loss_comparison(
            {"mlp": list(np.linspace(1, 0.2, 10)), "kan": list(np.linspace(1, 0.1, 10))},
            save_path=path,
        )
        assert os.path.isfile(path)
