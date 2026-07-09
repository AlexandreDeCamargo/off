"""OFF — orbital-free DFT with continuous normalizing flows."""

__version__ = "0.1.1"

from .functionals.functional import (
    Functional,
    CompositeFunctional,
    EnergyFunctional,
    FunctionalInputs,
    unit_coefficient,
)
from .functionals import (
    tf, weizsacker, tf_weizsacker,
    lda, b88, vwn, pw92, lda_b88,
    NuclearPotential, CoulombPotential,
    KatoCondition, HutcheonCuspCondition,
)
from .quadrature import (
    getGrid, get_grid, build_grid,
    grid_energy, grid_energy_from_checkpoint,
)
from .train.loss import build_energy_functional, create_loss_function
from .train.loop import training
from .ode_solver.eqx_ode import set_tolerances, get_tolerances
