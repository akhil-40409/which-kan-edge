import jax
import jax.numpy as jnp
import pytest
import time
from eigenflow.datasets import FeynmanDatasetGenerator, FEYNMAN_EQUATIONS

def test_all_equations_instantiation_and_generation():
    """Verify that all registered equations can be loaded and sampled from."""
    key = jax.random.PRNGKey(42)
    num_samples = 100
    
    for eq_id in FEYNMAN_EQUATIONS.keys():
        generator = FeynmanDatasetGenerator(eq_id)
        X, y, metadata = generator.generate(key, num_samples)
        
        assert X.shape == (num_samples, generator.info.dim)
        assert y.shape == (num_samples,)
        assert metadata["X_raw"].shape == (num_samples, generator.info.dim)
        assert metadata["y_raw"].shape == (num_samples,)
        assert not jnp.any(jnp.isnan(X))
        assert not jnp.any(jnp.isnan(y))

def test_input_scaling():
    """Test minmax_01, minmax_11, and standardize input scaling strategies."""
    key = jax.random.PRNGKey(123)
    num_samples = 500
    generator = FeynmanDatasetGenerator("I.12.2") # 4D inputs
    
    # 1. Raw
    X_raw, _, _ = generator.generate(key, num_samples, input_scaling="raw")
    # Verify values correspond to original domains
    for d, (d_min, d_max) in enumerate(generator.info.domains):
        assert jnp.all(X_raw[:, d] >= d_min - 1e-5)
        assert jnp.all(X_raw[:, d] <= d_max + 1e-5)
        
    # 2. MinMax 0 to 1
    X_minmax01, _, _ = generator.generate(key, num_samples, input_scaling="minmax_01")
    assert jnp.all(X_minmax01 >= -1e-6)
    assert jnp.all(X_minmax01 <= 1.0 + 1e-6)
    
    # 3. MinMax -1 to 1
    X_minmax11, _, _ = generator.generate(key, num_samples, input_scaling="minmax_11")
    assert jnp.all(X_minmax11 >= -1.0 - 1e-6)
    assert jnp.all(X_minmax11 <= 1.0 + 1e-6)
    
    # 4. Standardize
    X_std, _, _ = generator.generate(key, num_samples, input_scaling="standardize")
    means = jnp.mean(X_std, axis=0)
    stds = jnp.std(X_std, axis=0)
    
    for m in means:
        assert jnp.abs(m) < 1e-5
    for s in stds:
        assert jnp.abs(s - 1.0) < 1e-5

def test_target_scaling():
    """Test minmax_01, minmax_11, and standardize target scaling strategies."""
    key = jax.random.PRNGKey(456)
    num_samples = 500
    generator = FeynmanDatasetGenerator("I.12.1")
    
    # MinMax 0 to 1 target
    _, y_01, _ = generator.generate(key, num_samples, target_scaling="minmax_01")
    assert jnp.all(y_01 >= -1e-6)
    assert jnp.all(y_01 <= 1.0 + 1e-6)
    assert jnp.abs(jnp.min(y_01)) < 1e-5
    assert jnp.abs(jnp.max(y_01) - 1.0) < 1e-5
    
    # MinMax -1 to 1 target
    _, y_11, _ = generator.generate(key, num_samples, target_scaling="minmax_11")
    assert jnp.all(y_11 >= -1.0 - 1e-6)
    assert jnp.all(y_11 <= 1.0 + 1e-6)
    assert jnp.abs(jnp.min(y_11) + 1.0) < 1e-5
    assert jnp.abs(jnp.max(y_11) - 1.0) < 1e-5
    
    # Standardize target
    _, y_std, _ = generator.generate(key, num_samples, target_scaling="standardize")
    assert jnp.abs(jnp.mean(y_std)) < 1e-5
    assert jnp.abs(jnp.std(y_std) - 1.0) < 1e-5

def test_noise_models():
    """Test Gaussian, relative, and quantum shot noise injection."""
    key = jax.random.PRNGKey(789)
    num_samples = 1000
    generator = FeynmanDatasetGenerator("I.12.1") # mu * N
    
    # 1. No noise
    _, y_pure, meta_pure = generator.generate(key, num_samples, noise_type="none")
    assert jnp.allclose(y_pure, meta_pure["y_raw"])
    
    # 2. Gaussian noise
    noise_level = 0.1
    _, y_gauss, meta_gauss = generator.generate(key, num_samples, noise_level=noise_level, noise_type="gaussian")
    # Clean check: should differ from pure target
    assert not jnp.allclose(y_gauss, meta_gauss["y_raw"])
    diff_std = jnp.std(y_gauss - meta_gauss["y_raw"])
    expected_std = noise_level * jnp.std(meta_gauss["y_raw"])
    # Standard deviation of difference should be close to expected noise std
    assert jnp.abs(diff_std - expected_std) < 0.05
    
    # 3. Relative noise
    _, y_rel, meta_rel = generator.generate(key, num_samples, noise_level=noise_level, noise_type="relative")
    assert not jnp.allclose(y_rel, meta_rel["y_raw"])
    rel_diffs = (y_rel - meta_rel["y_raw"]) / (meta_rel["y_raw"] + 1e-8)
    assert jnp.abs(jnp.std(rel_diffs) - noise_level) < 0.05
    
    # 4. Quantum shot noise
    _, y_shot, meta_shot = generator.generate(key, num_samples, noise_level=noise_level, noise_type="quantum_shot")
    assert not jnp.allclose(y_shot, meta_shot["y_raw"])

def test_generation_speed_benchmark():
    """Verify that generation compiles under JAX and evaluates extremely fast."""
    key = jax.random.PRNGKey(101112)
    generator = FeynmanDatasetGenerator("I.43.16") # 5D variables
    num_samples = 100000
    
    # First call: triggers compilation
    t0 = time.time()
    _ = generator.generate(key, num_samples)
    t_compile = time.time() - t0
    
    # Second call: pure execution on compiled function
    t1 = time.time()
    _, _, _ = generator.generate(key, num_samples)
    t_exec = time.time() - t1
    
    print(f"\nCompile time: {t_compile:.4f}s")
    print(f"Execution time for {num_samples} samples: {t_exec:.4f}s")
    
    # Check that execution time is fast (usually < 5ms on CPU, but we set a conservative threshold of 100ms for test runners)
    assert t_exec < 0.1
