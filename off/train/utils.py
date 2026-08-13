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
        flow_model: Complete CNF model. Only its neural velocity field is trained.
        batch: Arguments passed to the loss function (often the static components
            of the model).
        optimizer: Optax optimizer.
        optimizer_state: Optimizer state.
        loss_fn: Loss function whose first arguments are the model and batch.

    Returns:
        tuple: (loss value, updated model, optimizer state)
    """
    # Nuclear positions and species descriptors are fixed inputs, not parameters.
    # Differentiate and optimize only the neural velocity field.
    trainable_flow = flow_model.flow.flow

    def trainable_loss(candidate_flow):
        candidate_model = eqx.tree_at(
            lambda model: model.flow.flow,
            flow_model,
            candidate_flow,
        )
        return loss_fn(candidate_model, batch, *loss_args)

    loss, grads = eqx.filter_value_and_grad(
        trainable_loss,
        has_aux=True,
    )(trainable_flow)

    updates, optimizer_state = optimizer.update(
        grads,
        optimizer_state,
        trainable_flow,
    )
    trainable_flow = eqx.apply_updates(trainable_flow, updates)
    flow_model = eqx.tree_at(
        lambda model: model.flow.flow,
        flow_model,
        trainable_flow,
    )
    
    return loss, flow_model, optimizer_state


@eqx.filter_jit
def eval_energy(flow_model, batch, loss_fn, *loss_args):
    """Forward-only energy evaluation on a batch (no gradients / no backprop).

    Same energy as ``loss_fn`` but never differentiated, so a much larger batch can be
    used cheaply for a lower-variance Monte-Carlo estimate. Returns ``(energy, F_values)``.
    """
    return loss_fn(flow_model, batch, *loss_args)
