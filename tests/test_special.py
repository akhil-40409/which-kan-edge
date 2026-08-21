import jax
import jax.numpy as jnp

from src.datasets import SPECIAL_FUNCTIONS, SpecialFunctionGenerator, make_dataset


def test_all_specials_finite(key):
    for fid in SpecialFunctionGenerator.available():
        gen = SpecialFunctionGenerator(fid)
        X, y, meta = gen.generate(key, 64)
        assert X.shape == (64, 1)
        assert y.shape == (64,)
        assert jnp.all(jnp.isfinite(X))
        assert jnp.all(jnp.isfinite(y))
        lo, hi = gen.info.domains[0]
        assert jnp.all(meta["X_raw"][:, 0] >= lo - 1e-6)
        assert jnp.all(meta["X_raw"][:, 0] <= hi + 1e-6)


def test_make_dataset_special(key):
    data = make_dataset("j0", key, n_samples=100)
    assert data["n_features"] == 1
    assert data["task_id"] == "j0"


def test_registry_nonempty():
    assert len(SPECIAL_FUNCTIONS) >= 5
