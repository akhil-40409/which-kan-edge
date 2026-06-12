# Running Relativistic Time Dilation MLP Experiment with Reusable Layer

import jax
import jax.numpy as jnp
import optax
import time
import matplotlib
matplotlib.use('Agg')  # Headless execution
from eigenflow.datasets import FeynmanDatasetGenerator
from eigenflow.layers import MLP
from eigenflow.utils import plot_regression_results

# Set up JAX keys
key = jax.random.PRNGKey(42)
dataset_key, init_key = jax.random.split(key)

# 1. Initialize dataset generator for Relativistic Time Dilation
generator = FeynmanDatasetGenerator("I.15.3t")

# 2. Generate 20,000 samples with input standardized scaling and 1% noise
X, y, metadata = generator.generate(
    key=dataset_key,
    num_samples=20000,
    noise_level=0.01,
    noise_type="gaussian",
    input_scaling="standardize",
    target_scaling="raw"
)

print(f"Generated {X.shape[0]} samples with {X.shape[1]} input variables.")
print(f"Formula: {metadata['formula']}")
print(f"Variables: {metadata['variables']}")

# 3. Model setup using our reusable MLP layer
layer_sizes = [3, 64, 64, 1]
mlp = MLP(layer_sizes, squeeze=True)
params = mlp.init(init_key)

# Setup Optimizer
lr = 3e-3
optimizer = optax.adam(lr)
opt_state = optimizer.init(params)

# Mean Squared Error Loss
def mse_loss(p, X_batch, y_batch):
    preds = mlp(p, X_batch)
    return jnp.mean((preds - y_batch) ** 2)

# Compiled update step
@jax.jit
def train_step(p, state, X_batch, y_batch):
    loss, grads = jax.value_and_grad(mse_loss)(p, X_batch, y_batch)
    updates, next_state = optimizer.update(grads, state, p)
    next_p = optax.apply_updates(p, updates)
    return next_p, next_state, loss

# Split train/test sets (80-20 split)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 4. Training Loop
epochs = 5000
losses = []
test_losses = []

print("Starting training...")
start_time = time.time()

for epoch in range(epochs):
    params, opt_state, loss = train_step(params, opt_state, X_train, y_train)
    losses.append(float(loss))
    
    if epoch % 500 == 0 or epoch == epochs - 1:
        test_loss = mse_loss(params, X_test, y_test)
        test_losses.append((epoch, float(test_loss)))
        print(f"Epoch {epoch:4d} | Train Loss: {loss:.6f} | Test Loss: {test_loss:.6f}")

end_time = time.time()
total_time = end_time - start_time
print(f"\nTraining completed in: {total_time:.4f} seconds!")
print(f"Time per epoch: {total_time/epochs*1000:.4f} milliseconds.")

# 5. Plot and Save Results
preds_test = mlp(params, X_test)
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
    model_label="JAX MLP Prediction",
    title="MLP Relativistic Time Dilation Regression (JAX)",
    save_path="experiments/MLPs/mlp_results.png"
)
