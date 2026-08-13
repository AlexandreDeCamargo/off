import jax
from jax import lax, numpy as jnp
import equinox as eqx
from jaxtyping import Array, Float, Int, Scalar

class _Flow(eqx.Module):
    linear_in: eqx.nn.Linear
    linear_out: eqx.nn.Linear
    blocks: list[eqx.nn.Linear]

    def __init__(
        self, 
        din: Int[Scalar, ""], 
        dim: Int[Scalar, ""], 
        key: jax.random.PRNGKey
    ):
        keys = jax.random.split(key, 5)
        
        self.linear_in = eqx.nn.Linear(din + 1, dim, key=keys[0])
        self.blocks = [eqx.nn.Linear(dim, dim, key=k) for k in jax.random.split(keys[1], 3)]
        linear_out = eqx.nn.Linear(dim, 1, key=keys[2])
        self.linear_out = eqx.tree_at(
            lambda layer: (layer.weight, layer.bias),
            linear_out,
            (jnp.zeros_like(linear_out.weight), jnp.zeros_like(linear_out.bias)),
        )

    def __call__(
        self, 
        t, 
        xi_norm, 
        zi_one_hot
    ):
        t = jnp.reshape(t, (1,))
        xi_norm = jnp.reshape(xi_norm, (1,))
        x = jnp.hstack((t, xi_norm, zi_one_hot)) 
        x = self.linear_in(x)  
        x = jnp.tanh(x)
        for block in self.blocks:
            x = block(x)  
            x = jnp.tanh(x)
        x = self.linear_out(x) 
        return x

class Radial_MLP(eqx.Module):
    xyz_nuclei: Array
    z_one_hot: Array
    flow: _Flow

    def __init__(
        self, 
        dim: int, 
        key, 
        xyz_nuclei, 
        z_one_hot
        ):
        din_flow = 1 + z_one_hot.shape[-1]
        self.xyz_nuclei = xyz_nuclei[:, None, :]
        self.z_one_hot = z_one_hot
        self.flow = _Flow(din_flow, dim, key=key)

    def __call__(self, states, t): 
        vmap_radial_block = jax.vmap(self.flow, in_axes=(None, 0, 0))
        z = lax.expand_dims(states, dimensions=(0,)) - self.xyz_nuclei
        z_norm = jnp.linalg.norm(z, axis=-1)
        x = vmap_radial_block(t, z_norm, self.z_one_hot)
        x = jnp.einsum('ijk,ij->k', z, x)
        return x

class CNF(eqx.Module):
    flow: Radial_MLP

    def __init__(
        self, 
        din: Int[Scalar, ""], 
        dim: Int[Scalar, ""], 
        mu: any, 
        one_hot:any, 
        key: jax.random.PRNGKey
    ):
        self.flow = Radial_MLP(dim, key, mu, one_hot)

    def __call__(self, states, t):
        data_dim = 3 #Hardcoded for 3 dimensions

        @jax.jit
        def _f_ode(self, states, t):
            x, log_px, score = states[:data_dim], states[data_dim:data_dim+1], states[data_dim+1:] 
            jac = jax.jacrev(self.flow)(x, t) 
            dtrJ = -1. * jnp.trace(jac)
            dz = self.flow(x, t)  
            return dz, dtrJ

        @jax.jit
        def f_ode(self,states,t):
            state, score = states[:-data_dim], states[-data_dim:]
            dx_and_dlopz, _f_vjp = jax.vjp(
                lambda state: _f_ode(self,state, t), state)
            dx, dlopz = dx_and_dlopz
            (vjp_all,) = _f_vjp((score, -1.))
            score_vjp, grad_div = vjp_all[:-1], vjp_all[-1]
            dscore = -score_vjp + grad_div
            return jnp.append(jnp.append(dx, dlopz), dscore)
        return f_ode(self, states, t)
