"""Special-function regression targets for symbolic / scientific benchmarks."""

from __future__ import annotations

from typing import Callable, Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy import special as sp


class SpecialInfo(NamedTuple):
    id: str
    name: str
    func: Callable[[jnp.ndarray], jnp.ndarray]
    dim: int
    domains: List[Tuple[float, float]]
    formula: str


def _np_fn(fn):
    """Wrap a NumPy/SciPy univariate fn → JAX-friendly (X[:,0] → y)."""

    def wrapped(X):
        x = np.asarray(X[:, 0], dtype=np.float64)
        return jnp.asarray(fn(x))

    return wrapped


def _legendre_p2(X):
    x = X[:, 0]
    return 0.5 * (3.0 * x**2 - 1.0)


def _legendre_p3(X):
    x = X[:, 0]
    return 0.5 * (5.0 * x**3 - 3.0 * x)


SPECIAL_FUNCTIONS: Dict[str, SpecialInfo] = {
    "j0": SpecialInfo(
        id="j0",
        name="Bessel J0",
        func=_np_fn(sp.j0),
        dim=1,
        domains=[(0.1, 10.0)],
        formula="J_0(x)",
    ),
    "j1": SpecialInfo(
        id="j1",
        name="Bessel J1",
        func=_np_fn(sp.j1),
        dim=1,
        domains=[(0.1, 10.0)],
        formula="J_1(x)",
    ),
    "y0": SpecialInfo(
        id="y0",
        name="Bessel Y0",
        func=_np_fn(sp.y0),
        dim=1,
        domains=[(0.5, 10.0)],
        formula="Y_0(x)",
    ),
    "erf": SpecialInfo(
        id="erf",
        name="Error function",
        func=lambda X: jax.lax.erf(X[:, 0]),
        dim=1,
        domains=[(-2.0, 2.0)],
        formula="erf(x)",
    ),
    "legendre_p2": SpecialInfo(
        id="legendre_p2",
        name="Legendre P2",
        func=_legendre_p2,
        dim=1,
        domains=[(-1.0, 1.0)],
        formula="(1/2)(3x^2 - 1)",
    ),
    "legendre_p3": SpecialInfo(
        id="legendre_p3",
        name="Legendre P3",
        func=_legendre_p3,
        dim=1,
        domains=[(-1.0, 1.0)],
        formula="(1/2)(5x^3 - 3x)",
    ),
    "airy_ai": SpecialInfo(
        id="airy_ai",
        name="Airy Ai",
        func=_np_fn(sp.airy),  # returns Ai only? airy returns tuple
        dim=1,
        domains=[(-5.0, 2.0)],
        formula="Ai(x)",
    ),
    "sinc": SpecialInfo(
        id="sinc",
        name="sinc (sin(x)/x)",
        func=lambda X: jnp.where(
            jnp.abs(X[:, 0]) < 1e-8, 1.0, jnp.sin(X[:, 0]) / X[:, 0]
        ),
        dim=1,
        domains=[(-8.0, 8.0)],
        formula="sin(x)/x",
    ),
}


# Fix Airy: scipy.special.airy returns (Ai, Aip, Bi, Bip)
def _airy_ai(X):
    x = np.asarray(X[:, 0], dtype=np.float64)
    ai, _, _, _ = sp.airy(x)
    return jnp.asarray(ai)


SPECIAL_FUNCTIONS["airy_ai"] = SpecialInfo(
    id="airy_ai",
    name="Airy Ai",
    func=_airy_ai,
    dim=1,
    domains=[(-5.0, 2.0)],
    formula="Ai(x)",
)


class SpecialFunctionGenerator:
    """Sample (X, y) for a registered special function."""

    def __init__(self, function_id: str):
        if function_id not in SPECIAL_FUNCTIONS:
            raise ValueError(
                f"Unknown special function {function_id!r}. "
                f"Available: {list(SPECIAL_FUNCTIONS)}"
            )
        self.info = SPECIAL_FUNCTIONS[function_id]

    def generate(
        self,
        key: jax.Array,
        num_samples: int,
        *,
        input_scaling: str = "minmax_11",
        target_scaling: str = "standardize",
        noise_level: float = 0.0,
    ) -> Tuple[jax.Array, jax.Array, dict]:
        keys = jax.random.split(key, self.info.dim + 2)
        cols = []
        for i, (lo, hi) in enumerate(self.info.domains):
            cols.append(
                jax.random.uniform(keys[i], (num_samples,), minval=lo, maxval=hi)
            )
        X_raw = jnp.stack(cols, axis=-1)
        y_raw = self.info.func(X_raw)
        if noise_level > 0:
            y_raw = y_raw + noise_level * jnp.std(y_raw) * jax.random.normal(
                keys[-2], y_raw.shape
            )

        x_mins = jnp.array([d[0] for d in self.info.domains])
        x_maxs = jnp.array([d[1] for d in self.info.domains])
        if input_scaling == "minmax_11":
            X = 2.0 * (X_raw - x_mins) / (x_maxs - x_mins + 1e-8) - 1.0
        elif input_scaling == "raw":
            X = X_raw
        else:
            X = (X_raw - jnp.mean(X_raw, axis=0)) / (jnp.std(X_raw, axis=0) + 1e-8)

        y_mean, y_std = jnp.mean(y_raw), jnp.std(y_raw) + 1e-8
        if target_scaling == "standardize":
            y = (y_raw - y_mean) / y_std
        elif target_scaling == "raw":
            y = y_raw
        else:
            y_min, y_max = jnp.min(y_raw), jnp.max(y_raw)
            y = 2.0 * (y_raw - y_min) / (y_max - y_min + 1e-8) - 1.0

        meta = {
            "X_raw": X_raw,
            "y_raw": y_raw,
            "name": self.info.name,
            "formula": self.info.formula,
            "next_key": keys[-1],
            "y_mean": y_mean,
            "y_std": y_std,
        }
        return X, y, meta

    @staticmethod
    def available() -> List[str]:
        return list(SPECIAL_FUNCTIONS.keys())
