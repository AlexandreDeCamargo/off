import jax
import jax.numpy as jnp
import jax.random as jrnd
import equinox as eqx
import optax
from optax import ema
import pandas as pd
import time
from typing import Optional

from ..flow.equiv_flows import CNF
from ..utils import one_hot_encode, coordinates, batch_generator, get_solver, get_scheduler
from ..promolecular.promolecular_dist import make_prior, ProMolecularDensity
from .utils import step, eval_energy
from .loss import create_loss_function, F_values
from ..config._config import Config

jax.config.update("jax_enable_x64", True)


def setup_molecule(mol_name: str, bond_length: float = 0.74144):
    """Setup molecular system."""
    Ne, atoms, z, coords = coordinates(mol_name, bond_length)
    mol = {'coords': coords, 'z': z}
    return Ne, atoms, z, coords, mol


def setup_model(coords, z, hidden_layer: int, key):
    """Initialize flow model."""
    mu = coords
    z_one_hot = one_hot_encode(z)
    data_dim = 3
    return CNF(data_dim, hidden_layer, mu, z_one_hot, key)


def setup_optimizer(flow_model, epochs: int, lr: float, scheduler_type: str):
    """Setup the optimizer for the neural velocity field only."""
    _lr = get_scheduler(epochs=epochs, sched_type=scheduler_type, lr=lr)
    optimizer = optax.adamw(_lr, weight_decay=1e-5)
    trainable_flow = flow_model.flow.flow
    optimizer_state = optimizer.init(eqx.filter(trainable_flow, eqx.is_inexact_array))
    return optimizer, optimizer_state


def setup_ema():
    """Setup EMA for tracking energies."""
    energies_ema = ema(decay=0.99)
    energies_state = energies_ema.init(
        F_values(energy=jnp.array(0.), kin=jnp.array(0.), 
                vnuc=jnp.array(0.), hart=jnp.array(0.), 
                xc=jnp.array(0.), cc=jnp.array(0.))
    )
    return energies_ema, energies_state


def log_metrics(itr: int, loss_epoch: float, losses: F_values, 
                energies_i_ema: F_values, elapsed_time: float):
    """Create metrics dictionaries for logging."""
    r_instant = {
        'epoch': itr,
        'E': loss_epoch - losses.cc,
        'T': losses.kin,
        'V': losses.vnuc,
        'H': losses.hart,
        'XC': losses.xc,
        'CC': losses.cc,
        't': elapsed_time
    }
    
    r_ema = {
        'epoch': itr,
        'E': energies_i_ema.energy - energies_i_ema.cc,
        'T': energies_i_ema.kin,
        'V': energies_i_ema.vnuc,
        'H': energies_i_ema.hart,
        'XC': energies_i_ema.xc,
        'CC': energies_i_ema.cc,
        't': elapsed_time
    }
    
    return r_instant, r_ema


def training(mol_name: str,
            bond_length: float = 1.4008538753,
            tw_kin: str = 'tf_w',
            lam: float = 1.0,
            n_pot: str = 'np',
            h_pot: str = 'coulomb',
            x_pot: str = 'lda',
            c_pot: str = 'vwn_c',
            cc_pot: str = 'kato', 
            batch_size: int = 256,
            hidden_layer: int = 64,
            epochs: int = 100,
            lr: float = 1e-5,
            scheduler_type: str = 'ones',
            solver_type: str = 'tsit5',
            prior_type: str = 'promolecular',
            prior_dist: Optional[ProMolecularDensity] = None,
            checkpoint_dir: str = './checkpoints',
            checkpoint_freq: int = 50,
            eval_every: int = 1000,
            eval_batch_size: Optional[int] = None,
            ): 
    """
    Main training loop.
    
    Parameters
    ----------
    mol_name : str
        Name of molecule
    bond_length: float 
        Bond length in a.u. 
    tw_kin : str
        Kinetic functional name
    n_pot : str
        External potential functional name
    h_pot : str
        Hartree functional name
    x_pot : str
        Exchange functional name
    c_pot : str
        Correlation functional name
    cc_pot : str
        Core correction functional name
    batch_size : int
        Batch size for training
    hidden_layer : int
        Hidden layer size for neural network
    epochs : int
        Number of training epochs
    lr : float
        Learning rate
    scheduler_type : str
        Type of learning rate scheduler
    solver_type : str
        ODE solver type
    prior_type: str
        Type of prior distribution for sampling  
    prior_dist : ProMolecularDensity, optional
        Initial distribution 
    checkpoint_dir : str
        Directory to save checkpoints
    checkpoint_freq : int
        Frequency of checkpoint saving
        
    Returns
    -------
    flow_model : CNF
        Trained flow model
    df : pd.DataFrame
        Training metrics
    df_ema : pd.DataFrame
        EMA training metrics
    """
    
    # Setup
    Ne, atoms, z, coords, mol = setup_molecule(mol_name, bond_length)
    
    key = jrnd.PRNGKey(0)
    _, key = jrnd.split(key)

    flow_model = setup_model(coords, z, hidden_layer, key)
    solver = get_solver(solver_type)
    optimizer, optimizer_state = setup_optimizer(flow_model, epochs, lr, scheduler_type)
    energies_ema, energies_state = setup_ema()
    sampling_dist = make_prior(prior_type, z, coords, Ne)

    gen_batches = batch_generator(key, batch_size, sampling_dist)
    
    grad_loss_fn = create_loss_function(
        kinetic_name=tw_kin,
        lam=lam,
        exchange_name=x_pot,
        correlation_name=c_pot,
        hartree_name=h_pot,
        external_name=n_pot,
        core_correction_name=cc_pot
    )
    
    # Training loop
    df = pd.DataFrame()
    df_ema = pd.DataFrame()

    # Forward-only eval: a larger batch (no backprop) for a lower-variance MC energy.
    eval_bs = eval_batch_size if eval_batch_size is not None else 10 * batch_size
    _, eval_key = jrnd.split(key)
    eval_batches = batch_generator(eval_key, eval_bs, sampling_dist)
    df_eval = pd.DataFrame()
    
    for itr in range(epochs + 1):
        start_time = time.time()
        
        batch = next(gen_batches)
        # batch = next(db_gen_batches)
        
        loss, flow_model, optimizer_state = step(
            flow_model, batch, optimizer, optimizer_state, 
            grad_loss_fn, solver, Ne, mol
        )
        
        elapsed_time = time.time() - start_time
        
        loss_epoch, losses = loss
        # Update EMA
        energies_i_ema, energies_state = energies_ema.update(losses, energies_state)
        
        # Log metrics
        r_instant, r_ema = log_metrics(itr, loss_epoch, losses, energies_i_ema, elapsed_time)
        
        df = pd.concat([df, pd.DataFrame([r_instant])], ignore_index=True)
        df_ema = pd.concat([df_ema, pd.DataFrame([r_ema])], ignore_index=True)
        
        print(f"Epoch {itr}: {r_ema}")
    
        df.to_csv(f"{Config.results_dir}/training_metrics.csv", index=False)
        df_ema.to_csv(f"{Config.results_dir}/training_metrics_ema.csv", index=False)
        
        # Save checkpoint
        if (itr % checkpoint_freq == 0
                or (itr >= max(0, epochs - 1000) and (epochs - itr) % 100 == 0)
                or itr == epochs):
            eqx.tree_serialise_leaves(f"{checkpoint_dir}/checkpoint_{itr}.eqx", flow_model)

        # Large-batch MC energy evaluation (forward only, no gradients)
        if eval_every and itr % eval_every == 0 and itr != 0:
            _, f_eval = eval_energy(
                flow_model, next(eval_batches), grad_loss_fn, solver, Ne, mol)
            r_eval = {
                'epoch': itr, 'batch': eval_bs,
                'E':  float(f_eval.energy - f_eval.cc),
                'T':  float(f_eval.kin),  'V':  float(f_eval.vnuc),
                'H':  float(f_eval.hart), 'XC': float(f_eval.xc), 'CC': float(f_eval.cc),
            }
            df_eval = pd.concat([df_eval, pd.DataFrame([r_eval])], ignore_index=True)
            df_eval.to_csv(f"{Config.results_dir}/eval_metrics.csv", index=False)
            print(f"  [eval @ {itr}]  E = {r_eval['E']:+.6f} Ha  "
                  f"(batch {eval_bs}, forward-only MC)")

    return flow_model, df, df_ema
