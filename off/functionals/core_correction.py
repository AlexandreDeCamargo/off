import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float
from jax import vmap, lax


class KatoCondition(eqx.Module):
    r"""
    Kato cusp-condition functional.

    Penalizes deviations from the correct electron-nucleus cusp behaviour, weighting
    the (Weizsacker-like) local kinetic term by a 1s-like envelope
    w_i(r) = exp(-a Z_i |r - R_i|) around each nucleus.

    Parameters
    ----------
    a : float, optional
        Envelope decay prefactor, by default 2/3.
    clip_cte : float, optional
        Small constant for numerical stability, by default 1e-5.
    """

    a: float
    clip_cte: float

    def __init__(self, a=2.0/3.0, clip_cte=1e-5):
        self.a = a
        self.clip_cte = clip_cte

    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch"]:
        r"""Uses x, score, Ne, mol; den, xp unused. Returns the cusp penalty per point."""
        def _wi(pts, molecule):
            r = jnp.sqrt(jnp.sum((pts - molecule['coords']) ** 2, axis=1) + self.clip_cte ** 2)
            return jnp.exp(-self.a * molecule['z'] * r)

        wi = vmap(_wi, in_axes=(None, 0), out_axes=-1)(x, mol)
        score_sqr = jnp.einsum('ij,ij->i', score, score)
        weizs = (1.0 / 8.0) * lax.expand_dims(score_sqr, (1,))
        kinetic = jnp.abs(weizs - (mol['z'] ** 2 / 2))
        return Ne * jnp.sum(kinetic * wi, axis=-1, keepdims=True)


class HutcheonCuspCondition(eqx.Module):
    r"""
    Exact nuclear-cusp cost functional.

    From Hutcheon & Wibowo-Teale, Phys. Rev. B 110, 195146 (2024), Eqs. (4-5, 7):

    E_{\text{CC}} = \sum_i C_i, \qquad
    C_i(\rho) = \int |\nabla \rho + 2 Z_i \rho\, \hat{r}_i|^2\, w_i(r)\, d^3 r,

    with  \hat{r}_i = (r - R_i)/|r - R_i|  and  w_i(r) = (\pi / Z_i^3)^{1/2} e^{-Z_i |r - R_i|}  [Eq. (5)].
    Since \nabla \rho = \rho\, s (s the score), the integrand is \rho^2 |s + 2 Z_i \hat{r}_i|^2,
    and the Kato cusp is satisfied exactly when s = -2 Z_i \hat{r}_i at R_i. With samples
    from p = \rho / N_e,

    C_i \approx N_e^2\, \mathbb{E}\left[ \rho(x)\, |s(x) + 2 Z_i \hat{r}_i(x)|^2\, w_i(x) \right].

    Parameters
    ----------
    clip_cte : float, optional
        Small constant for numerical stability, by default 1e-8.
    """

    clip_cte: float

    def __init__(self, clip_cte=1e-8):
        self.clip_cte = clip_cte

    def __call__(self, den, score, x, Ne, mol, xp) -> Float[Array, "batch 1"]:
        r"""Uses x, den, score, Ne, mol; xp unused. Returns the cusp cost per point."""
        coords, z = mol['coords'], mol['z']

        def per_atom(R_i, Z_i):
            r_vec = x - R_i
            r_norm = jnp.sqrt(jnp.sum(r_vec ** 2, axis=-1) + self.clip_cte ** 2)
            r_hat = r_vec / r_norm[:, None]
            w_i = jnp.sqrt(jnp.pi / Z_i ** 3) * jnp.exp(-Z_i * r_norm)
            cusp_sq = jnp.sum((score + 2.0 * Z_i * r_hat) ** 2, axis=-1)
            return cusp_sq * w_i

        total = jnp.sum(jax.vmap(per_atom)(coords, z), axis=0)
        return (Ne ** 2 * den.squeeze(-1) * total)[:, None]
