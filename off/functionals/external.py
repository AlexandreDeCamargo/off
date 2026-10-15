import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from jax import vmap


class NuclearPotential(eqx.Module):
    r"""
    External electron-nuclei attraction potential.

    V_{\text{ext}}(\boldsymbol{x}) = -N_e \sum_i \frac{Z_i}{|\boldsymbol{x} - \boldsymbol{R}_i|}

    Parameters
    ----------
    clip_cte : float, optional
        Small constant for numerical stability, by default 1e-5.
    """

    clip_cte: float

    def __init__(self, clip_cte=1e-5):
        self.clip_cte = clip_cte

    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch"]:
        r"""
        Parameters
        ----------
        x : Array
            Points where the potential is evaluated.
        Ne : int
            Number of electrons.
        mol : dict
            Nuclear coordinates and charges, {'coords': ..., 'z': ...}.

        Notes
        -----
        den, score, xp are accepted for the shared functional interface but unused here.

        Returns
        -------
        jax.Array
            Electron-nuclei attraction at each point (up to the rho factor).
        """
        def _potential(pts, molecule):
            r = jnp.sqrt(jnp.sum((pts - molecule['coords']) ** 2, axis=1) + self.clip_cte ** 2)
            return molecule['z'] / r

        r = vmap(_potential, in_axes=(None, 0), out_axes=-1)(x, mol)
        return -Ne * jnp.sum(r, axis=-1, keepdims=True)
