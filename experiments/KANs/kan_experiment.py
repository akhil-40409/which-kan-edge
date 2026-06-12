# Running Relativistic Time Dilation KAN Experiment with Reusable Layer

import jax
import jax.numpy as jnp
import optax
import time
import matplotlib
matplotlib.use('Agg')  # Headless execution
from eigenflow.datasets import FeynmanDatasetGenerator
from eigenflow.layers import KAN
from eigenflow.utils import plot_regression_results

# Set up JAX keys
key = jax.random.PRNGKey(42)
dataset_key, init_key = jax.random.split(key)

# 1. Initialize dataset generator for Relativistic Time Dilation
generator = FeynmanDatasetGenerator("I.15.3t")

# 2. Generate 20,000 samples scaled to [-1, 1] with 1% noise
X, y, metadata = generator.generate(
    key=dataset_key,
    num_samples=20000,
    noise_level=0.01,
    noise_type="gaussian",
    input_scaling="minmax_11",
    target_scaling="raw"
)

print(f"Generated {X.shape[0]} samples with {X.shape[1]} input variables.")
print(f"Formula: {metadata['formula']}")
print(f"Input ranges: {X.min()} to {X.max()}")

# 3. Model setup using our reusable KAN layer
layer_sizes = [3, 16, 16, 1]
kan = KAN(layer_sizes, grid_size=5, spline_order=3, grid_min=-1.0, grid_max=1.0)
params, grids = kan.init(init_key)

# Setup Optimizer
lr = 5e-3
optimizer = optax.adam(lr)
opt_state = optimizer.init(params)

# Mean Squared Error Loss
def mse_loss(p, g, X_batch, y_batch):
    preds = kan(p, g, X_batch)
    return jnp.mean((preds - y_batch) ** 2)

# Compiled update step
@jax.jit
def train_step(p, g, state, X_batch, y_batch):
    loss, grads = jax.value_and_grad(mse_loss, argnums=0)(p, g, X_batch, y_batch)
    updates, next_state = optimizer.update(grads, state, p)
    next_p = optax.apply_updates(p, updates)
    return next_p, next_state, loss

# Split train/test sets (80-20 split)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Training Loop
epochs = 3000
losses = []
test_losses = []

print("Starting training...")
start_time = time.time()

for epoch in range(epochs):
    params, opt_state, loss = train_step(params, grids, opt_state, X_train, y_train)
    losses.append(float(loss))
    
    if epoch % 500 == 0 or epoch == epochs - 1:
        test_loss = mse_loss(params, grids, X_test, y_test)
        test_losses.append((epoch, float(test_loss)))
        print(f"Epoch {epoch:4d} | Train Loss: {loss:.6f} | Test Loss: {test_loss:.6f}")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTraining completed in: {total_time:.4f} seconds!")
print(f"Time per epoch: {total_time/epochs*1000:.4f} milliseconds.")

# 5. Plot and Save Results
preds_test = kan(params, grids, X_test)
raw_test_inputs = metadata["X_raw"][split:]
raw_test_targets = metadata["y_raw"][split:]

plot_regression_results(
    losses=losses,
    y_true=y_test,
    y_pred=preds_test,
    test_losses=test_losses,
    X_raw=raw_test_inputs,
    y_raw=raw_test_targets,
    variables=metadata["variables"],
    slice_feature_idx=1,
    model_label="JAX KAN Prediction",
    title="KAN Relativistic Time Dilation Regression (JAX)",
    save_path="experiments/KANs/kan_results.png"
)
