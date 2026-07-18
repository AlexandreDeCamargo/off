import glob
import json
import re
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jrnd
import equinox as eqx
from pyscf import gto, dft

jax.config.update("jax_enable_x64", True)

from .flow.equiv_flows import CNF
from .ode_solver.eqx_ode import fwd_ode, rev_ode
from .utils import one_hot_encode, coordinates, get_solver
from .promolecular.promolecular_dist import make_prior
from .train.loss import build_energy_functional

AA_TO_BOHR = 1.8897259886


# ── model / prior loading ─────────────────────────────────────────────────────
def last_checkpoint(results_dir):
    """(path, epoch) of the highest-epoch checkpoint in results_dir/Checkpoints/."""
    ckpts = glob.glob(str(Path(results_dir) / "Checkpoints" / "checkpoint_*.eqx"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {results_dir}/Checkpoints/")
    ckpts.sort(key=lambda p: int(re.search(r'checkpoint_(\d+)\.eqx', p).group(1)))
    last = ckpts[-1]
    return last, int(re.search(r'checkpoint_(\d+)\.eqx', last).group(1))


def load_model(results_dir, p):
    """Rebuild the CNF for job_params `p` and load its last checkpoint."""
    Ne, atoms, z, coords = coordinates(p['mol_name'], p['bond_length'])
    model = CNF(din=3, dim=p['hidden_layer'], mu=coords,
                one_hot=one_hot_encode(z), key=jrnd.PRNGKey(0))
    ckpt, epoch = last_checkpoint(results_dir)
    model = eqx.tree_deserialise_leaves(ckpt, model)
    return model, get_solver(p['solver']), Ne, atoms, z, coords, epoch


def build_prior(p, z, coords, Ne):
    """Rebuild the base distribution used at training time (must match it)."""
    return make_prior(p.get('prior'), z, coords, Ne)


# ── grid construction (PySCF) ─────────────────────────────────────────────────
def _grids_from_mol(mol, level):
    """Build a PySCF Becke grid for `mol`; return (coords, weights) in Bohr."""
    grid = dft.gen_grid.Grids(mol)
    grid.level = level
    grid.build()
    return (jnp.asarray(grid.coords,  dtype=jnp.float64),
            jnp.asarray(grid.weights, dtype=jnp.float64))


def build_grid(atoms, coords, Ne, grid_level=3, basis="6-31G(d,p)", unit="B"):
    """PySCF molecular quadrature grid.  `coords` are interpreted in `unit`
    ('B'/'Bohr' or 'Angstrom'); the returned grid coords/weights are in Bohr."""
    atom_str = "; ".join(f"{a} {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}"
                         for a, c in zip(atoms, np.asarray(coords)))
    mol = gto.M(atom=atom_str, basis=basis, unit=unit, verbose=0, spin=int(Ne) % 2)
    return _grids_from_mol(mol, grid_level)


def get_grid(geom, level=3, *, units="angstrom", basis="6-31G(d,p)", spin=0):
    """User-facing quadrature grid.

    Parameters
    ----------
    geom : str
        Geometry in PySCF's ``atom=`` format — e.g. ``"H 0 0 0; H 0 0 0.74"``
        or a multi-line XYZ-style block.
    level : int
        PySCF grid level (the "grid size"); 
    units : {'angstrom', 'bohr'}
        Units the geometry is given in (PySCF's default is angstrom).
    basis, spin :
        Forwarded to ``pyscf.gto.M``.  The basis only sets the atom-centred
        grid partitioning; it does not affect the flow density.

    Returns
    -------
    (weights, coords) :
        Note the order, to match the listing ``w_grid, x_grid = get_grid(...)``;
        the internal :func:`build_grid` returns the opposite ``(coords, weights)``.
    """
    unit = "Bohr" if str(units).lower().startswith("b") else "Angstrom"
    mol = gto.M(atom=geom, basis=basis, unit=unit, spin=spin, verbose=0)
    coords, weights = _grids_from_mol(mol, level)
    return weights, coords


getGrid = get_grid   


def rho_on_grid(model, solver, prior, grid_coords, chunk=256):
    """Evaluate (positions, ρ_φ, score = ∇log ρ_φ) at the grid points."""
    x_l, rho_l, sc_l = [], [], []
    for i in range(0, grid_coords.shape[0], chunk):
        xc = grid_coords[i:i+chunk]; n = xc.shape[0]
        st_r = jnp.concatenate([xc, jnp.zeros((n, 1)), jnp.zeros((n, 3))], axis=1)
        zb, _ = rev_ode(model, st_r, solver)
        lp0 = prior.log_prob(zb)
        sc0 = prior.score(zb)
        xt, lpt, sct = fwd_ode(model, jnp.concatenate([zb, lp0, sc0], axis=1), solver)
        x_l.append(np.array(xt))
        rho_l.append(np.array(jnp.exp(lpt)).ravel())
        sc_l.append(np.array(sct))
    return np.concatenate(x_l), np.concatenate(rho_l), np.concatenate(sc_l)


def quadrature_energy(functional, x_np, rho_np, sc_np, grid_coords, grid_weights,
                      mol_dict, Ne, chunk=256, aux_coords=None, aux_rho_np=None,
                      aux_weights=None):
    """Integrate every energy term on the grid.

    The local terms use ``functional``'s component functionals.  The Hartree term is a
    real-space grid double sum (true 1/r), not the MC pairwise estimator.  If
    ``aux_coords``/``aux_rho_np``/``aux_weights`` are given it uses the TWO-GRID (offset)
    estimator — the main grid paired against a second Becke grid whose points never
    coincide, so no self-diagonal is dropped; otherwise it falls back to the single-grid
    (masked-diagonal) sum, which can under-count the near field.
    """
    w  = np.array(grid_weights)
    G  = rho_np.shape[0]
    rc = rho_np[:, None]
    measure = jnp.asarray(w * rho_np)        

    def _args(sl):
        return (jnp.array(rc[sl]), jnp.array(sc_np[sl]), jnp.array(x_np[sl]),
                Ne, mol_dict, None)

    def local(func):                         # ∫ f(...)·ρ dr  via the shared _integrate
        out = np.zeros(G)
        for i in range(0, G, chunk):
            sl = slice(i, min(i + chunk, G))
            out[sl] = np.array(func(*_args(sl))).ravel()
        return float(functional._integrate(jnp.asarray(out), measure))

    T    = local(functional.kinetic)
    E_X  = local(functional.exchange)
    E_C  = local(functional.correlation) if functional.correlation is not None else 0.0
    V_N  = local(functional.external)
    E_CC = local(functional.core_correction) if functional.core_correction is not None else 0.0

    # Hartree — grid double sum
    gc = np.array(grid_coords)
    vc = np.zeros(G)
    if aux_coords is not None:
        # TWO-GRID (offset) estimator: main grid (i) x auxiliary grid (j). The grids use
        # different Becke levels so their points never coincide -> 1/|r_i - r'_j| is never
        # singular and nothing is dropped, removing the single-grid self-interaction
        # under-count.   V_H = 0.5 Ne^2 * Σ_i w_i ρ_i Σ_j (w'_j ρ'_j)/|r_i - r'_j|
        gax      = np.array(aux_coords)
        aux_meas = np.array(aux_weights) * np.asarray(aux_rho_np)          # w'_j ρ'_j
        for i in range(0, G, chunk):
            xi = gc[i:i+chunk]
            r2 = np.sum((gax[None, :, :] - xi[:, None, :]) ** 2, axis=-1)  # (chunk, G_aux)
            vc[i:i+chunk] = np.dot(1. / np.sqrt(np.where(r2 == 0., np.inf, r2)), aux_meas)
    else:
        # Single grid vs itself, self-diagonal masked (can under-count the near field).
        for i in range(0, G, chunk):
            xi = gc[i:i+chunk]
            r2 = np.sum((gc[None, :, :] - xi[:, None, :]) ** 2, axis=-1)
            vc[i:i+chunk] = np.dot(1. / np.sqrt(np.where(r2 == 0., np.inf, r2)), w * rho_np)
    V_H = float(0.5 * Ne ** 2 * functional._integrate(jnp.asarray(vc), measure))

    # Nuclear repulsion 
    cn = np.array(mol_dict['coords']); zn = np.array(mol_dict['z']).ravel()
    E_NN = sum(zn[I] * zn[J] / float(np.linalg.norm(cn[I] - cn[J]))
               for I in range(len(cn)) for J in range(I + 1, len(cn)))

    return dict(T=T, V_N=V_N, V_H=V_H, E_X=E_X, E_C=E_C, E_CC=E_CC, E_NN=E_NN,
                E_total=T + V_N + V_H + E_X + E_C + E_CC + E_NN)


# ── high-level entry points ───────────────────────────────────────────────────
def grid_energy(model, prior, solver, coords, z, atoms, Ne, functional, *,
                grid_level=3, aux_grid_level=None, units="bohr", basis="6-31G(d,p)", chunk=256):
    """Build the grid, evaluate ρ_φ, and integrate all energy terms.

    Parameters
    ----------
    model, prior, solver : the trained CNF, its base distribution, ODE solver.
    coords, z, atoms, Ne : molecular geometry / charges / electron count.
    functional           : an EnergyFunctional (e.g. from build_energy_functional).
    grid_level           : PySCF grid level (the "grid size").
    aux_grid_level       : optional int.  If given, V_H uses the TWO-GRID (offset)
                           estimator — the main grid paired against a second Becke grid at
                           this level — avoiding the masked-diagonal under-count of the
                           single-grid Hartree.  None -> single-grid (masked) V_H.
    units                : 'bohr' or 'angstrom' — how `coords` are given; the flow
                           works in Bohr, so 'angstrom' inputs are converted.
    Returns the energy dict plus 'Ne_integral'.
    """
    coords = np.asarray(coords, dtype=float)
    unit = "Bohr" if str(units).lower().startswith("b") else "Angstrom"
    coords_bohr = coords if unit == "Bohr" else coords * AA_TO_BOHR
    gc, gw = build_grid(atoms, coords, Ne, grid_level=grid_level, basis=basis, unit=unit)
    x_np, rho_np, sc_np = rho_on_grid(model, solver, prior, gc, chunk=chunk)

    aux_coords = aux_rho = aux_w = None
    if aux_grid_level is not None:
        aux_coords, aux_w = build_grid(atoms, coords, Ne, grid_level=aux_grid_level,
                                       basis=basis, unit=unit)
        _, aux_rho, _ = rho_on_grid(model, solver, prior, aux_coords, chunk=chunk)

    mol_dict = {'coords': jnp.asarray(coords_bohr, dtype=jnp.float64), 'z': jnp.asarray(z)}
    en = quadrature_energy(functional, x_np, rho_np, sc_np, gc, gw, mol_dict, Ne, chunk=chunk,
                           aux_coords=aux_coords, aux_rho_np=aux_rho, aux_weights=aux_w)
    en['Ne_integral'] = float(np.dot(np.array(gw), Ne * rho_np))
    return en


def grid_energy_from_checkpoint(results_dir, *, grid_level=3, aux_grid_level=None,
                                basis="6-31G(d,p)", units="bohr", chunk=256,
                                recompute=False, cache=True):
    """One call from a trained run directory: read job_params.json (functional +
    geometry), load the last checkpoint, and integrate.  The geometry comes from
    ``coordinates`` (always Bohr), so no units flag is needed here.

    ``aux_grid_level`` (optional) switches V_H to the two-grid (offset) estimator; see
    :func:`grid_energy`.  The result is cached in ``results_dir/energy_summary.json``
    together with the grid setup — a cached summary is reused only when both
    ``grid_level`` and ``aux_grid_level`` match the request.  Pass ``recompute=True`` to
    ignore the cache, or ``cache=False`` to skip reading/writing it.
    """
    results_dir = Path(results_dir)
    summary_path = results_dir / "energy_summary.json"
    if cache and not recompute and summary_path.exists():
        with open(summary_path) as f:
            cached = json.load(f)
        if (cached.get("grid_level") == grid_level
                and cached.get("aux_grid_level") == aux_grid_level):
            return cached
        # different grid setup (or a pre-existing summary lacking these keys) -> recompute

    with open(results_dir / "job_params.json") as f:
        p = json.load(f)
    model, solver, Ne, atoms, z, coords, epoch = load_model(results_dir, p)
    prior = build_prior(p, z, coords, Ne)
    functional = build_energy_functional(
        kinetic_name=p['kinetic'], lam=p['lam'], exchange_name=p['exchange'],
        correlation_name=p['correlation'], hartree_name=p['hartree'],
        external_name=p['external'], core_correction_name=p['core_correction'],
    )
    en = grid_energy(model, prior, solver, coords, z, atoms, Ne, functional,
                     grid_level=grid_level, aux_grid_level=aux_grid_level,
                     units=units, basis=basis, chunk=chunk)
    en.update(epoch=epoch, mol_name=p['mol_name'], bond_length=p['bond_length'],
              grid_level=grid_level, aux_grid_level=aux_grid_level)
    if cache:
        with open(summary_path, "w") as f:
            json.dump(en, f, indent=4)
    return en


def _print_energy(results_dir, en):
    print(f"\n{results_dir}")
    grid = f"grid={en.get('grid_level')}"
    if en.get('aux_grid_level') is not None:
        grid += f"+aux{en.get('aux_grid_level')}"
    print(f"  mol={en.get('mol_name')}  R={en.get('bond_length')}  "
          f"epoch={en.get('epoch')}  [{grid}]")
    print("  " + "-" * 36)
    for k in ("T", "V_N", "V_H", "E_X", "E_C", "E_CC", "E_NN"):
        print(f"  {k:8s} = {en[k]:+.6f} Ha")
    print("  " + "-" * 36)
    print(f"  {'E_total':8s} = {en['E_total']:+.6f} Ha")
    print(f"  {'N_e':8s} = {en['Ne_integral']:.4f}   (∫ρ, should be Ne)")


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Grid (quadrature) energy of a trained OFF run directory.")
    ap.add_argument("results_dir", nargs="+",
                    help="bl_* run dir(s) with job_params.json and Checkpoints/ "
                         "(shell globs like Results/H2/<method>/bl_* are fine)")
    ap.add_argument("--grid_level", type=int, default=1, help="PySCF grid level")
    ap.add_argument("--aux_grid_level", type=int, default=None,
                    help="auxiliary Becke grid level for a TWO-GRID (offset) Hartree "
                         "estimator (avoids the single-grid masked-diagonal under-count); "
                         "omit for the single-grid V_H")
    ap.add_argument("--bs", type=int, default=256, help="grid chunk size")
    ap.add_argument("--basis", type=str, default="6-31G(d,p)",
                    help="PySCF basis (sets grid partitioning only)")
    ap.add_argument("--recompute", action="store_true",
                    help="ignore cached energy_summary.json and recompute")
    args = ap.parse_args()

    for rd in args.results_dir:
        en = grid_energy_from_checkpoint(
            rd, grid_level=args.grid_level, aux_grid_level=args.aux_grid_level,
            basis=args.basis, chunk=args.bs, recompute=args.recompute)
        _print_energy(rd, en)


if __name__ == "__main__":
    main()
