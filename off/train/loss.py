import jax.numpy as jnp
from typing import NamedTuple
from ..functionals.kinetic import tf, weizsacker, tf_weizsacker
from ..functionals.exchange_correlation import lda, b88, vwn, pw92, lda_b88
from ..functionals.hartree import CoulombPotential
from ..functionals.external import NuclearPotential
from ..functionals.core_correction import KatoCondition, HutcheonCuspCondition
from ..functionals.functional import FunctionalInputs, EnergyFunctional
from ..ode_solver.eqx_ode import fwd_ode
import jax
class F_values(NamedTuple):
    """Container for energy components."""
    energy: float
    kin: float
    vnuc: float
    hart: float
    xc: float
    cc: float

FUNCTIONAL_CLASSES = {
    # Kinetic
    'tf': lambda: tf,
    'w': lambda: weizsacker(),
    'tf_w': lambda: tf_weizsacker(),

    # Exchange
    'lda': lambda: lda,
    'b88_x': lambda: b88,
    'lda_b88_x': lambda: lda_b88(),

    # Correlation
    'vwn_c': lambda: vwn,
    'pw92_c': lambda: pw92,

    # Hartree
    'coulomb': CoulombPotential,        

    # External
    'np': NuclearPotential,

    # Core correction
    'kato': KatoCondition,
    'hutcheon': HutcheonCuspCondition,
}

def _build_kinetic(kinetic_name: str, lam: float):
    """Build the kinetic functional."""
    if kinetic_name == 'w':
        return weizsacker(lam)
    if kinetic_name == 'tf_w':
        return tf_weizsacker(lam)
    return tf


def build_energy_functional(
    kinetic_name: str = 'tf',
    lam: float = 1.0,
    exchange_name: str = 'lda',
    correlation_name: str = 'none',
    hartree_name: str = 'coulomb',
    external_name: str = 'np',
    core_correction_name: str = 'none',
):
    return EnergyFunctional(
        kinetic=_build_kinetic(kinetic_name, lam),
        external=FUNCTIONAL_CLASSES[external_name](),
        hartree=FUNCTIONAL_CLASSES[hartree_name](),
        exchange=FUNCTIONAL_CLASSES[exchange_name](),
        correlation=FUNCTIONAL_CLASSES[correlation_name]()
                    if correlation_name != 'none' else None,
        core_correction=FUNCTIONAL_CLASSES[core_correction_name]()
                    if core_correction_name != 'none' else None,
    )


def create_loss_function(
    kinetic_name: str = 'tf',
    lam: float = 1.0,
    exchange_name: str = 'lda',
    correlation_name: str = 'none',
    hartree_name: str = 'coulomb',
    external_name: str = 'np',
    core_correction_name: str = 'none'
):
    """
    Factory function to create a loss function with specific functionals.

    Parameters
    ----------
    kinetic_name : str
        Name of kinetic functional ('tf', 'w', 'tf_w')
    lam : float
        Weizsäcker prefactor λ in TF-λW 
    exchange_name : str
        Name of exchange functional ('lda', 'b88_x')
    correlation_name : str
        Name of correlation functional ('vwn_c', 'pw92_c', 'none')
    hartree_name : str
        Name of Hartree functional ('coulomb')
    external_name : str
        Name of external potential functional ('np')
    core_correction_name : str
        Name of core correction functional ('kato', 'hutcheon', 'none')

    Returns
    -------
    grad_loss : callable
    """
    functional = build_energy_functional(
        kinetic_name, lam, exchange_name, correlation_name,
        hartree_name, external_name, core_correction_name,
    )

    def grad_loss(model, z_and_logpz, solver, Ne, mol):
        """
        Compute the loss function.
        """
        x, log_px, _score = fwd_ode(model, z_and_logpz, solver)

        bs = int(x.shape[0] / 2)

        den_all, x_all, score_all = jnp.exp(log_px), x, _score
        score, scorep = score_all[:bs], score_all[bs:]
        den,   denp   = den_all[:bs],   den_all[bs:]
        x,     xp     = x_all[:bs],     x_all[bs:]

        inp   = FunctionalInputs(den=den, score=score, x=x, Ne=Ne, mol=mol, xp=xp)
        terms = functional.terms(inp)
        t_e, n_e, h_e = terms["kin"], terms["vnuc"], terms["hart"]
        x_e, c_e, cc_e = terms["x"], terms["c"], terms["cc"]
        xc_e = x_e + c_e

        e = t_e + n_e + h_e + xc_e + cc_e
        energy = functional._integrate(jnp.reshape(e, (-1,)), 1.0 / bs)

        f_values = F_values(
            energy=energy,
            kin=jnp.mean(t_e),
            vnuc=jnp.mean(n_e),
            hart=jnp.mean(h_e),
            xc=jnp.mean(xc_e),
            cc=jnp.mean(cc_e) if isinstance(cc_e, jnp.ndarray) else cc_e
        )

        return energy, f_values

    return grad_loss
