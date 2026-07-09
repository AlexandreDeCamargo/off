"""Utility functions for training."""
from collections.abc import Callable

import equinox as eqx
import optax
from jaxtyping import Array, PyTree, Scalar

@eqx.filter_jit
def step(
    flow_model: PyTree,
    batch: Array,  
    optimizer: optax.GradientTransformation,
    optimizer_state: PyTree,
    loss_fn: Callable[[PyTree, PyTree], Scalar],
    *loss_args 
    ):
    """Carry out a training step.

    Args:
        params: Flow model
        batch: Arguments passed to the loss function (often the static components
            of the model).
        optimizer: Optax optimizer.
        optimizer_state: Optimizer state.
        loss_fn: The loss function. This should take params and static as the first two
            arguments.

    Returns:
        tuple: (loss_val, params, optmizer state)
    """
    # Compute loss and gradients
    loss, grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(flow_model, batch,*loss_args)
    
    # Update the model parameters
    updates, optimizer_state = optimizer.update(grads, optimizer_state,flow_model)
    flow_model = eqx.apply_updates(flow_model, updates)
    
    return loss, flow_model, optimizer_state


@eqx.filter_jit
def eval_energy(flow_model, batch, loss_fn, *loss_args):
    """Forward-only energy evaluation on a batch (no gradients / no backprop).

    Same energy as ``loss_fn`` but never differentiated, so a much larger batch can be
    used cheaply for a lower-variance Monte-Carlo estimate. Returns ``(energy, F_values)``.
    """
    return loss_fn(flow_model, batch, *loss_args)