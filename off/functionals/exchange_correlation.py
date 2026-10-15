import jax.numpy as jnp
from jax import lax
from .functional import Functional, CompositeFunctional, unit_coefficient


def lda_eps(den, score, x, Ne, mol, xp):
    r"""
    Local density approximation (LDA) / Dirac exchange functional.

    See eq. 2.72 in "Time-Dependent Density-Functional Theory", Carsten A. Ullrich.

    E_{\text{X}}^{\text{LDA}}[\rho] = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3} \int \rho(\boldsymbol{x})^{4/3} d\boldsymbol{x}
                                    = -\frac{3}{4}\left(\frac{3}{\pi}\right)^{1/3} N_e^{4/3}\, \mathbb{E}_{\rho_\phi}\left[ \rho_\phi(\boldsymbol{x})^{1/3} \right]

    Parameters
    ----------
    den : Array
        Density.
    Ne : int
        Number of electrons.

    Notes
    -----
    score, x, mol, xp are accepted for the shared functional interface but unused here.

    Returns
    -------
    jax.Array
        LDA exchange energy density (up to the rho factor).
    """
    l = -(3 / 4) * (Ne ** (4 / 3)) * (3 / jnp.pi) ** (1 / 3)
    return l * den ** (1 / 3)


def b88_eps(den, score, x, Ne, mol, xp, clip_cte=1e-30, beta=0.0042):
    r"""
    B88 exchange functional.

    See eq. 8 in https://journals.aps.org/pra/abstract/10.1103/PhysRevA.38.3098
    See also https://github.com/ElectronicStructureLibrary/libxc/blob/master/maple/gga_exc/gga_x_b88.mpl

    Spin-unpolarised (closed-shell) limit of Becke's eq. 8, which is a sum over spin
    channels: setting \rho_\sigma = \rho/2 gives X_\sigma = 2^{1/3} X and

    E_{\text{X}}^{\text{B88}}[\rho] = -2^{1/3}\beta \int \frac{X^2}{1 + 6\beta\, 2^{1/3} X \sinh^{-1}(2^{1/3} X)} \rho(\boldsymbol{x})^{4/3} d\boldsymbol{x},
    \qquad X = \frac{|\nabla \rho|}{\rho^{4/3}}.

    Parameters
    ----------
    den : Array
        Density.
    score : Array
        Gradient of the log-density, s = (nabla rho)/rho.
    Ne : int
        Number of electrons.
    clip_cte : float, optional
        Small constant for numerical stability, by default 1e-30.
    beta : float, optional
        B88 parameter, by default 0.0042.

    Notes
    -----
    x, mol, xp are accepted for the shared functional interface but unused here.

    Returns
    -------
    jax.Array
        B88 exchange energy density (up to the rho factor).
    """
    den_clipped = jnp.clip(den, clip_cte)
    log_den = jnp.log2(den_clipped)
    score_sqr = jnp.einsum('ij,ij->i', score, score)
    grad_den_norm_sq = lax.expand_dims(score_sqr, (1,)) * den_clipped * den_clipped
    log_grad_den_norm = jnp.log2(jnp.clip(grad_den_norm_sq, clip_cte)) / 2
    log_x_sigma = log_grad_den_norm - 4 / 3.0 * log_den
    x_phys = 2 ** (log_x_sigma - jnp.log2(Ne) / 3.0)
    x_cs = 2 ** (1 / 3.0) * x_phys
    b88_e = -(beta * 2 ** (1 / 3.0 + log_den / 3 + 2 * log_x_sigma
                           - jnp.log2(1 + 6 * beta * x_cs * jnp.arcsinh(x_cs))))
    return b88_e * Ne ** (2 / 3)


def vwn_eps(den, score, x, Ne, mol, xp, clip_cte=1e-30):
    r"""
    VWN correlation functional.

    See original paper eq. 4.4 in https://cdnsciencepub.com/doi/abs/10.1139/p80-159
    See also the text after eq. 8.9.6.1 in https://www.theoretical-physics.com/dev/quantum/dft.html

    \epsilon_{\text{C}}^{\text{VWN}} = \frac{A}{2}\left\{ \ln\frac{y^2}{Y(y)}
        + \frac{2b}{Q}\arctan\frac{Q}{2y+b}
        - \frac{b y_0}{Y(y_0)}\left[ \ln\frac{(y-y_0)^2}{Y(y)}
        + \frac{2(b+2y_0)}{Q}\arctan\frac{Q}{2y+b}\right] \right\}

    Parameters
    ----------
    den : Array
        Density.
    Ne : int
        Number of electrons.
    clip_cte : float, optional
        Small constant for numerical stability, by default 1e-30.

    Notes
    -----
    score, x, mol, xp are accepted for the shared functional interface but unused here.

    Returns
    -------
    jax.Array
        VWN correlation energy density (up to the rho factor).
    """
    A, b, c, x0 = 0.0621814, 3.72744, 12.9352, -0.10498
    den_clipped = jnp.where(den > clip_cte, den, 0.0)
    log_den = jnp.log2(jnp.clip(den_clipped, clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - (log_den + jnp.log2(Ne)) / 3.0
    log_x = log_rs / 2
    x_ = 2. ** log_x
    X = 2. ** (2. * log_x) + 2. ** (log_x + jnp.log2(b)) + c
    X0 = x0 ** 2 + b * x0 + c
    Q = jnp.sqrt(4 * c - b ** 2)
    e_PF = A / 2. * (
        2. * jnp.log(x_) - jnp.log(X) + 2. * b / Q * jnp.arctan(Q / (2. * x_ + b))
        - b * x0 / X0 * (jnp.log((x_ - x0) ** 2. / X)
                         + 2. * (2. * x0 + b) / Q * jnp.arctan(Q / (2. * x_ + b)))
    )
    return Ne * e_PF


def pw92_eps(den, score, x, Ne, mol, xp, clip_cte=1e-30):
    r"""
    PW92 correlation functional.

    See eq. 10 in https://journals.aps.org/prb/abstract/10.1103/PhysRevB.45.13244

    \epsilon_{\text{C}}^{\text{PW92}} = -2A(1 + \alpha_1 r_s)
        \ln\left[ 1 + \frac{1}{2A(\beta_1 r_s^{1/2} + \beta_2 r_s + \beta_3 r_s^{3/2} + \beta_4 r_s^2)} \right]

    Parameters
    ----------
    den : Array
        Density.
    Ne : int
        Number of electrons.
    clip_cte : float, optional
        Small constant for numerical stability, by default 1e-30.

    Notes
    -----
    score, x, mol, xp are accepted for the shared functional interface but unused here.

    Returns
    -------
    jax.Array
        PW92 correlation energy density (up to the rho factor).
    """
    A_, alpha1 = 0.031091, 0.21370
    beta1, beta2, beta3, beta4 = 7.5957, 3.5876, 1.6382, 0.49294
    log_den = jnp.log2(jnp.clip(den, clip_cte))
    log_rs = jnp.log2((3 / (4 * jnp.pi)) ** (1 / 3)) - (log_den + jnp.log2(Ne)) / 3.0
    brs_1_2 = 2 ** (log_rs / 2 + jnp.log2(beta1))
    ars = 2 ** (log_rs + jnp.log2(alpha1))
    brs = 2 ** (log_rs + jnp.log2(beta2))
    brs_3_2 = 2 ** (3 * log_rs / 2 + jnp.log2(beta3))
    brs2 = 2 ** (2 * log_rs + jnp.log2(beta4))
    e_PF = -2 * A_ * (1 + ars) * jnp.log(1 + (1 / (2 * A_)) / (brs_1_2 + brs + brs_3_2 + brs2))
    return Ne * e_PF


lda  = Functional(coefficients=unit_coefficient, energy_per_particle=lda_eps)
b88  = Functional(coefficients=unit_coefficient, energy_per_particle=b88_eps)
vwn  = Functional(coefficients=unit_coefficient, energy_per_particle=vwn_eps)
pw92 = Functional(coefficients=unit_coefficient, energy_per_particle=pw92_eps)


def lda_b88():
    r"""LDA + B88 exchange."""
    return CompositeFunctional(lda, b88)
