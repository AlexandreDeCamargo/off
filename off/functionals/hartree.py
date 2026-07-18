import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float

class CoulombPotential(eqx.Module):
    r"""
    Classical electron-electron repulsion (Hartree) potential, all-pairs estimator.

    V_{\text{Hartree}} = \frac{1}{2} N_e^2 \left\langle \frac{1}{|\boldsymbol{x}_i - \boldsymbol{x}'_j|} \right\rangle_j

    Parameters
    ----------
    clip_cte : float, optional
        Small constant for numerical stability, by default 1e-5.
    """

    clip_cte: float

    def __init__(self, clip_cte=1e-5):
        self.clip_cte = clip_cte

    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch 1"]:
        x2 = jnp.sum(x * x, axis=-1)
        xp2 = jnp.sum(xp * xp, axis=-1)
        r2 = x2[:, None] + xp2[None, :] - 2.0 * (x @ xp.T)
        r2 = jnp.maximum(r2, 0.0) + self.clip_cte
        v = jnp.mean(1.0 / jnp.sqrt(r2), axis=-1, keepdims=True)
        return 0.5 * (Ne ** 2) * v


class CoulombPotential_element_wise(eqx.Module):
    r"""
    Classical electron-electron repulsion (Hartree) potential, element-wise pairs.

    Pairs each x_i with x'_i (the two half-batches), giving `batch` Monte-Carlo
    pairs:

    V_{\text{Hartree}}(\boldsymbol{x}, \boldsymbol{x}') = \frac{1}{2} N_e^2 \frac{1}{|\boldsymbol{x} - \boldsymbol{x}'|}

    Parameters
    ----------
    clip_cte : float, optional
        Small constant for numerical stability, by default 1e-5.
    """

    clip_cte: float

    def __init__(self, clip_cte=1e-5):
        self.clip_cte = clip_cte

    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch 1"]:
        z = jnp.sum((x - xp) ** 2 + self.clip_cte, axis=-1, keepdims=True)
        return 0.5 * (Ne ** 2) / jnp.sqrt(z)
