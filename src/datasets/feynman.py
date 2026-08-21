import jax
import jax.numpy as jnp
from typing import Dict, Tuple, NamedTuple, Callable, List, Optional
from functools import partial

class EquationInfo(NamedTuple):
    id: str
    name: str
    func: Callable[[jnp.ndarray], jnp.ndarray]
    dim: int
    domains: List[Tuple[float, float]]
    variables: List[str]
    formula: str

# --- AI Feynman Equations Registry ---
# These represent a subset of the equations from the AI Feynman database
# (https://arxiv.org/abs/1905.11481) with mathematically safe domains.

def eq_I_6_20a(X):
    # Gaussian density: f(sigma, theta) = (1 / (sqrt(2*pi)*sigma)) * exp(-theta^2 / (2*sigma^2))
    sigma = X[:, 0]
    theta = X[:, 1]
    return (1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)) * jnp.exp(- (theta ** 2) / (2.0 * sigma ** 2))

def eq_I_8_14(X):
    # Distance: f(x1, x2, y1, y2) = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    x1, x2, y1, y2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return jnp.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def eq_I_12_1(X):
    # Friction force: f(mu, N) = mu * N
    mu, N = X[:, 0], X[:, 1]
    return mu * N

def eq_I_12_2(X):
    # Coulomb's law: f(q1, q2, epsilon, r) = q1 * q2 / (4 * pi * epsilon * r^2)
    q1, q2, epsilon, r = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (q1 * q2) / (4.0 * jnp.pi * epsilon * r**2)

def eq_I_15_3t(X):
    # Relativistic time dilation: f(t, v, c) = t / sqrt(1 - v^2/c^2)
    t, v, c = X[:, 0], X[:, 1], X[:, 2]
    ratio = jnp.clip(v / c, 0.0, 0.95)
    return t / jnp.sqrt(1.0 - ratio**2)

def eq_I_18_4(X):
    # Center of mass: f(m1, r1, m2, r2) = (m1*r1 + m2*r2) / (m1 + m2)
    m1, r1, m2, r2 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (m1 * r1 + m2 * r2) / (m1 + m2)

def eq_I_30_5(X):
    # Two-slit diffraction/grating: f(theta, n, phi) = theta * sin^2(n*phi/2) / sin^2(phi/2)
    theta, n, phi = X[:, 0], X[:, 1], X[:, 2]
    denom = jnp.sin(phi / 2.0) ** 2
    num = jnp.sin(n * phi / 2.0) ** 2
    return theta * (num / denom)

def eq_I_32_17(X):
    # Larmor power formula: f(q, a, epsilon, c) = q^2 * a^2 / (6 * pi * epsilon * c^3)
    q, a, epsilon, c = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (q**2 * a**2) / (6.0 * jnp.pi * epsilon * c**3)

def eq_I_43_16(X):
    # Drift velocity: f(mu, q, E, tau, m) = mu * q * E * tau / m
    mu, q, E, tau, m = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    return (mu * q * E * tau) / m

def eq_II_4_24(X):
    # Dipole potential: f(p, theta, epsilon, r) = p * cos(theta) / (4 * pi * epsilon * r^2)
    p, theta, epsilon, r = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return (p * jnp.cos(theta)) / (4.0 * jnp.pi * epsilon * r**2)

def eq_II_11_27(X):
    # Polarizability: f(N, alpha) = N * alpha / (1 - N * alpha / 3)
    N, alpha = X[:, 0], X[:, 1]
    val = N * alpha
    return val / (1.0 - val / 3.0)

def eq_III_4_32(X):
    # Planck distribution term: f(h, nu, kB, T) = h * nu / (exp(h*nu / (kB*T)) - 1)
    h, nu, kB, T = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    exponent = (h * nu) / (kB * T)
    exponent = jnp.clip(exponent, 0.05, 10.0)
    return (h * nu) / (jnp.exp(exponent) - 1.0)

FEYNMAN_EQUATIONS: Dict[str, EquationInfo] = {
    "I.6.20a": EquationInfo(
        id="I.6.20a",
        name="Gaussian Density",
        func=eq_I_6_20a,
        dim=2,
        domains=[(1.0, 3.0), (-5.0, 5.0)],
        variables=["sigma", "theta"],
        formula="1 / (sqrt(2*pi)*sigma) * exp(-theta^2 / (2*sigma^2))"
    ),
    "I.8.14": EquationInfo(
        id="I.8.14",
        name="Distance Formula",
        func=eq_I_8_14,
        dim=4,
        domains=[(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
        variables=["x1", "x2", "y1", "y2"],
        formula="sqrt((x2 - x1)^2 + (y2 - y1)^2)"
    ),
    "I.12.1": EquationInfo(
        id="I.12.1",
        name="Friction Force",
        func=eq_I_12_1,
        dim=2,
        domains=[(0.0, 1.0), (0.0, 10.0)],
        variables=["mu", "N"],
        formula="mu * N"
    ),
    "I.12.2": EquationInfo(
        id="I.12.2",
        name="Coulomb's Law",
        func=eq_I_12_2,
        dim=4,
        domains=[(-5.0, 5.0), (-5.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        variables=["q1", "q2", "epsilon", "r"],
        formula="q1 * q2 / (4 * pi * epsilon * r^2)"
    ),
    "I.15.3t": EquationInfo(
        id="I.15.3t",
        name="Relativistic Time Dilation",
        func=eq_I_15_3t,
        dim=3,
        domains=[(1.0, 10.0), (0.0, 0.9), (1.0, 2.0)],
        variables=["t", "v", "c"],
        formula="t / sqrt(1 - v^2/c^2)"
    ),
    "I.18.4": EquationInfo(
        id="I.18.4",
        name="Center of Mass",
        func=eq_I_18_4,
        dim=4,
        domains=[(1.0, 5.0), (-5.0, 5.0), (1.0, 5.0), (-5.0, 5.0)],
        variables=["m1", "r1", "m2", "r2"],
        formula="(m1*r1 + m2*r2) / (m1 + m2)"
    ),
    "I.30.5": EquationInfo(
        id="I.30.5",
        name="Two-slit Diffraction",
        func=eq_I_30_5,
        dim=3,
        domains=[(0.0, 10.0), (1.0, 5.0), (0.5, 2.5)],
        variables=["theta", "n", "phi"],
        formula="theta * sin^2(n*phi/2) / sin^2(phi/2)"
    ),
    "I.32.17": EquationInfo(
        id="I.32.17",
        name="Larmor Power Formula",
        func=eq_I_32_17,
        dim=4,
        domains=[(0.0, 5.0), (0.0, 5.0), (1.0, 5.0), (1.0, 5.0)],
        variables=["q", "a", "epsilon", "c"],
        formula="q^2 * a^2 / (6 * pi * epsilon * c^3)"
    ),
    "I.43.16": EquationInfo(
        id="I.43.16",
        name="Drift Velocity",
        func=eq_I_43_16,
        dim=5,
        domains=[(0.5, 2.0), (0.5, 2.0), (0.5, 2.0), (0.5, 2.0), (1.0, 5.0)],
        variables=["mu", "q", "E", "tau", "m"],
        formula="mu * q * E * tau / m"
    ),
    "II.4.24": EquationInfo(
        id="II.4.24",
        name="Dipole Potential",
        func=eq_II_4_24,
        dim=4,
        domains=[(1.0, 5.0), (0.0, jnp.pi), (1.0, 5.0), (1.0, 5.0)],
        variables=["p", "theta", "epsilon", "r"],
        formula="p * cos(theta) / (4 * pi * epsilon * r^2)"
    ),
    "II.11.27": EquationInfo(
        id="II.11.27",
        name="Polarizability",
        func=eq_II_11_27,
        dim=2,
        domains=[(0.1, 1.0), (0.1, 1.0)],
        variables=["N", "alpha"],
        formula="N * alpha / (1 - N * alpha / 3)"
    ),
    "III.4.32": EquationInfo(
        id="III.4.32",
        name="Planck Distribution Term",
        func=eq_III_4_32,
        dim=4,
        domains=[(1.0, 2.0), (1.0, 2.0), (1.0, 2.0), (1.0, 5.0)],
        variables=["h", "nu", "kB", "T"],
        formula="h * nu / (exp(h*nu / (kB*T)) - 1)"
    )
}

# --- Vectorized JIT Compiler Pipeline ---

@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5, 6))
def _generate_pipeline(
    key: jax.random.PRNGKey,
    num_samples: int,
    domains: Tuple[Tuple[float, float], ...],
    func: Callable[[jnp.ndarray], jnp.ndarray],
    input_scaling: str,
    target_scaling: str,
    noise_type: str,
    noise_level: float
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray], jax.random.PRNGKey]:
    
    # 1. Split keys for uniform sampling, target noise, etc.
    keys = jax.random.split(key, len(domains) + 2)
    
    # 2. Sample inputs within variable domains
    samples = []
    for i, (d_min, d_max) in enumerate(domains):
        samples.append(jax.random.uniform(keys[i], shape=(num_samples,), minval=d_min, maxval=d_max))
    X_raw = jnp.stack(samples, axis=-1)
    
    # 3. Evaluate the target equation
    y_raw = func(X_raw)
    
    # 4. Inject noise
    noise_key = keys[-2]
    y_noisy = y_raw
    
    if noise_type == "gaussian":
        y_std = jnp.std(y_raw) + 1e-8
        noise = jax.random.normal(noise_key, shape=y_raw.shape) * noise_level * y_std
        y_noisy = y_raw + noise
    elif noise_type == "relative":
        noise = jax.random.normal(noise_key, shape=y_raw.shape) * noise_level
        y_noisy = y_raw * (1.0 + noise)
    elif noise_type == "quantum_shot":
        # Quantum expectation values lie in [-1, 1]. We scale y_raw to [-1, 1] internally to
        # calculate standard quantum shot variance: Var = (1 - <O>^2) / shots.
        shots = 1.0 / (noise_level ** 2 + 1e-8)
        y_min_val = jnp.min(y_raw)
        y_max_val = jnp.max(y_raw)
        y_range = y_max_val - y_min_val + 1e-8
        
        y_scaled_temp = 2.0 * (y_raw - y_min_val) / y_range - 1.0
        y_scaled_temp = jnp.clip(y_scaled_temp, -1.0, 1.0)
        
        variance = (1.0 - y_scaled_temp ** 2) / shots
        std_dev = jnp.sqrt(jnp.clip(variance, 0.0, None))
        
        noise = std_dev * jax.random.normal(noise_key, shape=y_raw.shape)
        y_noisy_scaled = y_scaled_temp + noise
        y_noisy_scaled = jnp.clip(y_noisy_scaled, -1.0, 1.0)
        
        y_noisy = (y_noisy_scaled + 1.0) * 0.5 * y_range + y_min_val

    # 5. Scale inputs
    x_mins = jnp.array([d[0] for d in domains])
    x_maxs = jnp.array([d[1] for d in domains])
    x_means = jnp.mean(X_raw, axis=0)
    x_stds = jnp.std(X_raw, axis=0) + 1e-8
    
    if input_scaling == "minmax_01":
        X = (X_raw - x_mins) / (x_maxs - x_mins + 1e-8)
    elif input_scaling == "minmax_11":
        X = 2.0 * (X_raw - x_mins) / (x_maxs - x_mins + 1e-8) - 1.0
    elif input_scaling == "standardize":
        X = (X_raw - x_means) / x_stds
    else: # "raw"
        X = X_raw
        
    # 6. Scale targets
    y_mean = jnp.mean(y_noisy)
    y_std = jnp.std(y_noisy) + 1e-8
    y_min = jnp.min(y_noisy)
    y_max = jnp.max(y_noisy)
    
    if target_scaling == "minmax_01":
        y = (y_noisy - y_min) / (y_max - y_min + 1e-8)
    elif target_scaling == "minmax_11":
        y = 2.0 * (y_noisy - y_min) / (y_max - y_min + 1e-8) - 1.0
    elif target_scaling == "standardize":
        y = (y_noisy - y_mean) / y_std
    else: # "raw"
        y = y_noisy
        
    stats = {
        "x_min": x_mins,
        "x_max": x_maxs,
        "x_mean": x_means,
        "x_std": x_stds,
        "y_min": y_min,
        "y_max": y_max,
        "y_mean": y_mean,
        "y_std": y_std
    }
    
    return X, y, X_raw, y_raw, y_noisy, stats, keys[-1]

# --- Public API Interface ---

class FeynmanDatasetGenerator:
    """JAX-accelerated dataset generator for symbolic regression benchmarks.
    
    Supports highly optimal, parallel data generation based on selected equations from
    the AI Feynman database. Tailored for benchmarking MLPs, KANs, and QKANs.
    """
    
    def __init__(self, equation_id: str):
        """Initializes the generator for the specified equation.
        
        Args:
            equation_id: The ID of the equation to generate datasets for (e.g., 'I.12.2').
        """
        if equation_id not in FEYNMAN_EQUATIONS:
            raise ValueError(f"Equation ID '{equation_id}' not found in registry. "
                             f"Available equations: {list(FEYNMAN_EQUATIONS.keys())}")
        self.equation_info = FEYNMAN_EQUATIONS[equation_id]
        
    @property
    def info(self) -> EquationInfo:
        """Returns metadata about the registered equation."""
        return self.equation_info
        
    def generate(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        noise_level: float = 0.0,
        noise_type: str = "none",
        input_scaling: str = "raw",
        target_scaling: str = "raw"
    ) -> Tuple[jax.Array, jax.Array, Dict]:
        """Generates datasets in parallel using compiled JAX routines.
        
        Args:
            key: PRNG key for random generation.
            num_samples: Number of samples to generate.
            noise_level: Magnitude of noise to inject (standard deviation or relative level).
            noise_type: Noise strategy ('none', 'gaussian', 'relative', 'quantum_shot').
            input_scaling: Scaling strategy for inputs ('raw', 'minmax_01', 'minmax_11', 'standardize').
            target_scaling: Scaling strategy for targets ('raw', 'minmax_01', 'minmax_11', 'standardize').
            
        Returns:
            X: Matrix of generated inputs (shape [num_samples, input_dim])
            y: Vector of target values (shape [num_samples])
            metadata: Helper dictionary containing raw unscaled tensors, scaler stats, and metadata.
        """
        valid_scalings = {"raw", "minmax_01", "minmax_11", "standardize"}
        if input_scaling not in valid_scalings:
            raise ValueError(f"Invalid input_scaling: {input_scaling}. Options: {valid_scalings}")
        if target_scaling not in valid_scalings:
            raise ValueError(f"Invalid target_scaling: {target_scaling}. Options: {valid_scalings}")
            
        valid_noises = {"none", "gaussian", "relative", "quantum_shot"}
        if noise_type not in valid_noises:
            raise ValueError(f"Invalid noise_type: {noise_type}. Options: {valid_noises}")
            
        domains_tuple = tuple(self.equation_info.domains)
        
        X, y, X_raw, y_raw, y_noisy, stats, next_key = _generate_pipeline(
            key,
            num_samples,
            domains_tuple,
            self.equation_info.func,
            input_scaling,
            target_scaling,
            noise_type,
            noise_level
        )
        
        metadata = {
            "X_raw": X_raw,
            "y_raw": y_raw,
            "y_noisy_raw": y_noisy,
            "scaling_stats": stats,
            "next_key": next_key,
            "formula": self.equation_info.formula,
            "variables": self.equation_info.variables,
            "name": self.equation_info.name
        }
        
        return X, y, metadata

    @staticmethod
    def get_available_equations() -> List[Dict[str, str]]:
        """Returns a list of all available equations and their formulas."""
        return [
            {
                "id": eq.id,
                "name": eq.name,
                "dim": str(eq.dim),
                "formula": eq.formula,
                "variables": ", ".join(eq.variables)
            }
            for eq in FEYNMAN_EQUATIONS.values()
        ]

    def generate_splits(
        self,
        key: jax.random.PRNGKey,
        num_samples: int,
        *,
        train_frac: float = 0.7,
        val_frac: float = 0.15,
        noise_level: float = 0.0,
        noise_type: str = "none",
        input_scaling: str = "minmax_11",
        target_scaling: str = "standardize",
    ) -> Dict[str, jax.Array]:
        """Generate data and return a shuffled train/val/test dict."""
        from src.datasets import train_val_test_split

        k_data, k_split = jax.random.split(key)
        X, y, _ = self.generate(
            k_data,
            num_samples,
            noise_level=noise_level,
            noise_type=noise_type,
            input_scaling=input_scaling,
            target_scaling=target_scaling,
        )
        return train_val_test_split(
            X, y, k_split, train_frac=train_frac, val_frac=val_frac
        )
