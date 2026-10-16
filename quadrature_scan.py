"""
Grid-quadrature total energy for every bond length of a molecule.

For the chosen molecule it walks **all** method directories and, inside each,
every bl_* subdirectory.  For each bond length it loads the LAST saved
checkpoint and runs the same deterministic grid integration as scan_pes.py
( E_total = T + V_N + V_H + E_X + E_C + E_CC + E_NN ).  Everything is written
to a single CSV associated with that molecule.

The Hartree term V_H uses a TWO-GRID (offset) estimator: the main grid
(--grid_level) is paired against a coarser auxiliary grid (--aux_grid_level,
default grid_level - 1).  Because the two grids' Becke points never coincide,
1/|r - r'| is never singular and NO self term has to be dropped — this avoids
the systematic under-count of the single-grid (masked-diagonal) estimator.

For molecules (not single atoms) it ALSO grid-integrates the constituent
single atoms under the *same* method tag (e.g. N2 -> N at bl_0.00), and
adds the atomic reference and the binding energy to the CSV.

Directory layout assumed (same as main.py):
    Results/{mol}/{method}/bl_X.XXXX/
        Checkpoints/checkpoint_*.eqx
        job_params.json
    Results/{atom}/{method}/bl_0.00/   (used for the binding reference)
Results/ is located next to this script, so it runs from anywhere.

Usage
-----
# scan H2 (per-molecule flag):
python quadrature_scan.py --H2

# several molecules at once, or via --mol:
python quadrature_scan.py --H2 --N2
python quadrature_scan.py --mol H2 H10

# choose the auxiliary Hartree grid level explicitly:
python quadrature_scan.py --N2 --grid_level 3 --aux_grid_level 2 --recompute

# force re-integration (ignore cached energy_summary.json):
python quadrature_scan.py --H10 --recompute

Output (one CSV per molecule, written under Results/{mol}/):
    Results/{mol}/quadrature_{mol}.csv
      columns: method, R_bohr, epoch, E_total, T, V_N, V_H,
               E_X, E_C, E_CC, E_NN, Ne_int, E_atoms, dE_bind_Ha
      (E_atoms / dE_bind_Ha present only for molecules whose constituent
       atoms were trained under the same method tag.)

Note: cached energy_summary.json files are reused only when their stored
grid_level AND aux_grid_level match the current run.  Summaries written by the
old single-grid version (which stored neither) are recomputed automatically.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc, jax
import argparse
import glob
import json
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jrnd
import equinox as eqx
import numpy as np
import pandas as pd
from pyscf import gto, dft

jax.config.update("jax_enable_x64", True)

from flow.equiv_flows import CNF
from ode_solver.eqx_ode import fwd_ode, rev_ode
from utils import one_hot_encode, coordinates, get_solver
from promolecular.promolecular_dist import ProMolecularDensity, AtomDBDistribution
from train.loss import FUNCTIONAL_CLASSES, _build_kinetic

# Results/ sits next to this script (of_flows/Results), independent of CWD.
_SCRIPT_DIR = Path(__file__).resolve().parent

# Molecules understood by utils.coordinates() — each gets a --<name> flag.
KNOWN_MOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
              "H2", "N2", "O2", "F2", "HF", "CO", "LiH", "H10"]

# Single-nucleus systems — no binding energy / no atomic reference for these.
SINGLE_ATOMS = {"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    allow_abbrev=False)  # so --H is not treated as a prefix of --H2 / --H10
for _m in KNOWN_MOLS:
    parser.add_argument(f"--{_m}", action="store_true",
                        help=f"Scan molecule {_m}")
parser.add_argument("--mol", type=str, nargs="+", default=[], metavar="NAME",
                    help="Molecule name(s) to scan (alternative to the flags)")
parser.add_argument("--results_root", type=str, default=None,
                    help="Override Results root (default: <script_dir>/Results)")
parser.add_argument("--bs", type=int, default=256, help="Grid chunk size")
parser.add_argument("--grid_level", type=int, default=3, help="PySCF grid level (main grid)")
parser.add_argument("--aux_grid_level", type=int, default=None,
                    help="PySCF grid level for the auxiliary Hartree grid "
                         "(offset grid; default: grid_level - 1). The Hartree "
                         "double sum pairs the main grid against this one.")
parser.add_argument("--recompute", action="store_true",
                    help="Re-run grid integration even if energy_summary.json is cached")
parser.add_argument("--out", type=str, default=None,
                    help="Output CSV path (default: Results/{mol}/quadrature_{mol}.csv)")
args = parser.parse_args()

# Auxiliary grid level for the Hartree term (offset from the main grid).
AUX_GRID_LEVEL = (args.aux_grid_level if args.aux_grid_level is not None
                  else max(0, args.grid_level - 1))

# ── grid-integration machinery (identical to scan_pes.py / atom_energies.py) ──

def last_checkpoint(results_dir: Path):
    """Return (path, epoch) of the highest-epoch checkpoint available."""
    ckpts = glob.glob(str(results_dir / "Checkpoints" / "checkpoint_*.eqx"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {results_dir}/Checkpoints/")
    ckpts.sort(key=lambda p: int(re.search(r'checkpoint_(\d+)\.eqx', p).group(1)))
    last  = ckpts[-1]
    epoch = int(re.search(r'checkpoint_(\d+)\.eqx', last).group(1))
    return last, epoch


def load_model(results_dir: Path, p: dict):
    Ne, atoms, z, coords = coordinates(p['mol_name'], p['bond_length'])
    z_one_hot = one_hot_encode(z)
    key   = jrnd.PRNGKey(0)
    model = CNF(din=3, dim=p['hidden_layer'], mu=coords, one_hot=z_one_hot, key=key)
    ckpt, epoch = last_checkpoint(results_dir)
    model = eqx.tree_deserialise_leaves(ckpt, model)
    return model, get_solver(p['solver']), Ne, atoms, z, coords, epoch


def build_prior(p, z, coords, Ne):
    prior = ProMolecularDensity(z.ravel(), coords)
    return prior


def rho_on_grid(model, solver, prior, grid_coords, chunk):
    x_l, rho_l, sc_l = [], [], []
    for i in range(0, grid_coords.shape[0], chunk):
        xc = grid_coords[i:i+chunk]; n = xc.shape[0]
        st_r = jnp.concatenate([xc, jnp.zeros((n,1)), jnp.zeros((n,3))], axis=1)
        zb, _ = rev_ode(model, st_r, solver)
        lp0   = prior.log_prob(zb)
        sc0   = prior.score(zb)
        xt, lpt, sct = fwd_ode(model,
            jnp.concatenate([zb, lp0, sc0], axis=1), solver)
        x_l.append(np.array(xt))
        rho_l.append(np.array(jnp.exp(lpt)).ravel())
        sc_l.append(np.array(sct))
    return np.concatenate(x_l), np.concatenate(rho_l), np.concatenate(sc_l)


def compute_energy(p, x_np, rho_np, sc_np, grid_coords, grid_weights, mol_dict, Ne, chunk,
                   aux_coords, aux_rho_np, aux_weights):
    w   = np.array(grid_weights)
    G   = rho_np.shape[0]
    rc  = rho_np[:, None]

    t_func  = _build_kinetic(p['kinetic'], p['lam'])
    x_func  = FUNCTIONAL_CLASSES[p['exchange']]()
    n_func  = FUNCTIONAL_CLASSES[p['external']]()
    c_func  = FUNCTIONAL_CLASSES[p['correlation']]()   if p['correlation']    != 'none' else None
    cc_func = FUNCTIONAL_CLASSES[p['core_correction']]() if p['core_correction']!= 'none' else None

    t_e=np.zeros(G); x_e=np.zeros(G); n_e=np.zeros(G); c_e=np.zeros(G); cc_e=np.zeros(G)
    for i in range(0, G, chunk):
        sl = slice(i, min(i+chunk, G))
        den=jnp.array(rc[sl]); sc=jnp.array(sc_np[sl]); xc=jnp.array(x_np[sl])
        t_e[sl]  = np.array(t_func(den, sc, Ne)).ravel()
        x_e[sl]  = np.array(x_func(den, sc, Ne)).ravel()
        n_e[sl]  = np.array(n_func(xc, Ne, mol_dict)).ravel()
        if c_func:  c_e[sl]  = np.array(c_func(den, sc, Ne)).ravel()
        if cc_func: cc_e[sl] = np.array(cc_func(xc, den, sc, Ne, mol_dict)).ravel()

    T   = float(np.dot(w, t_e  * rho_np))
    E_X = float(np.dot(w, x_e  * rho_np))
    V_N = float(np.dot(w, n_e  * rho_np))
    E_C = float(np.dot(w, c_e  * rho_np))
    E_CC= float(np.dot(w, cc_e * rho_np))

    # Hartree — TWO-GRID (offset) double sum: main grid (i) × auxiliary grid (j).
    # The two grids use different Becke levels, so their points don't coincide and
    # 1/|r_i - r'_j| is never singular: nothing is dropped (no diagonal mask), which
    # removes the self-interaction under-count of the single-grid estimator.
    #     V_H = 0.5 * Ne^2 * Σ_i Σ_j  w_i ρ_i  /|r_i - r'_j|  w'_j ρ'_j
    gcoords  = np.array(grid_coords)
    gax      = np.array(aux_coords)
    aux_meas = np.array(aux_weights) * aux_rho_np            # w'_j ρ'_j  (inner measure)
    vc = np.zeros(G)
    for i in range(0, G, chunk):
        xi = gcoords[i:i+chunk]
        r2 = np.sum((gax[None, :, :] - xi[:, None, :]) ** 2, axis=-1)   # (chunk, G_aux)
        # guard only triggers if a main and aux point exactly coincide (offset grids: ~never)
        vc[i:i+chunk] = np.dot(1. / np.sqrt(np.where(r2 == 0., np.inf, r2)), aux_meas)
    V_H = float(0.5 * Ne**2 * np.dot(w * rho_np, vc))

    # Nuclear repulsion (== 0 for a single atom)
    cn = np.array(mol_dict['coords']); zn = np.array(mol_dict['z']).ravel()
    E_NN = sum(zn[I]*zn[J]/float(np.linalg.norm(cn[I]-cn[J]))
               for I in range(len(cn)) for J in range(I+1, len(cn)))

    return dict(T=T, V_N=V_N, V_H=V_H, E_X=E_X, E_C=E_C, E_CC=E_CC,
                E_NN=E_NN, E_total=T+V_N+V_H+E_X+E_C+E_CC+E_NN)


def analyse_one(results_dir: Path, chunk: int, grid_level: int,
                aux_grid_level: int, recompute: bool) -> dict:
    """Grid energy for one bl_X.XXXX directory (cached in energy_summary.json)."""
    summary_path = results_dir / "energy_summary.json"
    if summary_path.exists() and not recompute:
        with open(summary_path) as f:
            cached = json.load(f)
        # Reuse the cache only if it was produced with the SAME grid setup.
        # Old single-grid summaries (and runs at different levels) lack/≠ these
        # keys, so they are recomputed automatically instead of silently reused.
        if (cached.get("grid_level") == grid_level
                and cached.get("aux_grid_level") == aux_grid_level):
            print(f"  {results_dir.name}: loading cached energy_summary.json")
            return cached
        print(f"  {results_dir.name}: cached summary uses a different grid setup — recomputing")

    print(f"  {results_dir.name}: running analysis ...")
    with open(results_dir / "job_params.json") as f:
        p = json.load(f)

    model, solver, Ne, atoms, z, coords, epoch = load_model(results_dir, p)
    print(f"    checkpoint epoch = {epoch}")
    mol_dict = {'coords': coords, 'z': z}
    prior    = build_prior(p, z, coords, Ne)

    atom_str = "; ".join(f"{a} {c[0]:.8f} {c[1]:.8f} {c[2]:.8f}"
                         for a, c in zip(atoms, coords))
    mol_pyscf = gto.M(atom=atom_str, basis="6-31G(d,p)", unit="B",
                      verbose=0, spin=int(Ne)%2)

    # Main integration grid.
    grid = dft.gen_grid.Grids(mol_pyscf)
    grid.level = grid_level; grid.build()
    gcoords = jnp.array(grid.coords,  dtype=jnp.float64)
    gw      = jnp.array(grid.weights, dtype=jnp.float64)

    # Auxiliary (offset) grid used only for the Hartree inner integral.
    aux_grid = dft.gen_grid.Grids(mol_pyscf)
    aux_grid.level = aux_grid_level; aux_grid.build()
    ga_coords = jnp.array(aux_grid.coords,  dtype=jnp.float64)
    ga_w      = jnp.array(aux_grid.weights, dtype=jnp.float64)

    x_np, rho_np, sc_np = rho_on_grid(model, solver, prior, gcoords, chunk)
    _, aux_rho_np, _    = rho_on_grid(model, solver, prior, ga_coords, chunk)
    print(f"    main grid: {gcoords.shape[0]} pts (level {grid_level})  |  "
          f"aux grid: {ga_coords.shape[0]} pts (level {aux_grid_level})")

    Ne_int  = float(np.dot(np.array(gw), Ne * rho_np))
    pos_err = float(np.max(np.abs(x_np - np.array(gcoords))))
    print(f"    ∫ρdx={Ne_int:.5f} (should be {Ne})  |  round-trip err={pos_err:.2e}")

    en = compute_energy(p, x_np, rho_np, sc_np, gcoords, gw, mol_dict, Ne, chunk,
                        aux_coords=ga_coords, aux_rho_np=aux_rho_np, aux_weights=ga_w)
    result = {**en, 'bond_length': p['bond_length'], 'epoch': epoch,
              'mol_name': p['mol_name'], 'Ne_integral': Ne_int,
              'grid_level': grid_level, 'aux_grid_level': aux_grid_level}

    with open(summary_path, "w") as f:
        json.dump(result, f, indent=4)
    return result


# ── molecule selection ────────────────────────────────────────────────────────
selected = list(args.mol)
for m in KNOWN_MOLS:
    if getattr(args, m):
        selected.append(m)
# de-duplicate, keep order
selected = list(dict.fromkeys(selected))

if not selected:
    parser.error("No molecule selected. Use a flag (e.g. --H2) or --mol H2 [...].")

root = Path(args.results_root).resolve() if args.results_root else (_SCRIPT_DIR / "Results")
print(f"Results root : {root}")
print(f"Hartree grids: main level {args.grid_level}  ×  aux level {AUX_GRID_LEVEL}\n")


def constituents(mol: str) -> dict:
    """{element: count} from a formula, e.g. N2->{N:2}, HF->{H:1,F:1}, H10->{H:10}."""
    out = {}
    for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", mol):
        if el:
            out[el] = out.get(el, 0) + (int(n) if n else 1)
    return out


def atom_reference(method_name: str, mol: str, chunk, grid_level, aux_grid_level, recompute):
    """Grid energy reference Σ_atoms count·E(atom) for a molecule, using the
    SAME method tag (Results/{atom}/{method}/bl_0.00).

    Returns (E_atoms, {element: E_atom}) or (None, None) if any constituent
    atom directory is missing.
    """
    total = 0.0
    detail = {}
    for el, n in constituents(mol).items():
        adir = root / el / method_name / "bl_0.00"
        if not (adir / "job_params.json").exists():
            print(f"  atom reference: {el} not found at {adir} — binding skipped")
            return None, None
        try:
            data = analyse_one(adir, chunk, grid_level, aux_grid_level, recompute)
        except Exception as e:
            print(f"  atom reference: {el} FAILED — {e}")
            return None, None
        detail[el] = data['E_total']
        total += n * data['E_total']
    return total, detail


def scan_molecule(mol: str):
    """Integrate every method × bond length for one molecule; write its CSV.
    For non-atoms, also integrate the constituent atom(s) and report binding."""
    mol_dir = root / mol
    if not mol_dir.is_dir():
        print(f"[{mol}] SKIP — {mol_dir} not found\n")
        return

    is_atom = mol in SINGLE_ATOMS
    method_dirs = sorted(d for d in mol_dir.iterdir() if d.is_dir())
    rows = []
    for method_dir in method_dirs:
        bl_dirs = sorted(method_dir.glob("bl_*"),
                         key=lambda d: float(d.name.split("_")[1]))
        if not bl_dirs:
            continue
        print(f"[{mol}] method: {method_dir.name}  ({len(bl_dirs)} bond lengths)")

        # Single-atom reference for the binding energy (same method tag, once per method).
        E_atoms = None
        if not is_atom:
            E_atoms, detail = atom_reference(method_dir.name, mol,
                                             args.bs, args.grid_level, AUX_GRID_LEVEL,
                                             args.recompute)
            if E_atoms is not None:
                ref = "  ".join(f"{n}*E({el})={detail[el]:+.6f}"
                                for el, n in constituents(mol).items())
                print(f"  atom reference (grid): {ref}  ->  Σ = {E_atoms:+.6f} Ha")

        for bl_dir in bl_dirs:
            if not (bl_dir / "job_params.json").exists():
                print(f"  {bl_dir.name}: missing job_params.json — skipping")
                continue
            try:
                data = analyse_one(bl_dir, args.bs, args.grid_level, AUX_GRID_LEVEL,
                                   args.recompute)
            except Exception as e:
                print(f"  {bl_dir.name}: FAILED — {e}")
                continue
            row = {
                "method":  method_dir.name,
                "R_bohr":  data['bond_length'],
                "epoch":   data.get('epoch', '?'),
                "E_total": data['E_total'],
                "T":       data['T'],
                "V_N":     data['V_N'],
                "V_H":     data['V_H'],
                "E_X":     data['E_X'],
                "E_C":     data.get('E_C', 0.0),
                "E_CC":    data.get('E_CC', 0.0),
                "E_NN":    data['E_NN'],
                "Ne_int":  data['Ne_integral'],
            }
            if E_atoms is not None:
                row["E_atoms"]    = E_atoms
                # ΔE = Σ E(atom) - E(molecule)   (positive = bound)
                row["dE_bind_Ha"] = E_atoms - data['E_total']
            rows.append(row)
            msg = (f"    R={data['bond_length']:.4f} Bohr  epoch={data.get('epoch','?'):>6}"
                   f"  E_total={data['E_total']:+.6f} Ha")
            if E_atoms is not None:
                msg += f"  ΔE={E_atoms - data['E_total']:+.6f} Ha"
            print(msg)
            jax.clear_caches()
            gc.collect()
        print()

    if not rows:
        print(f"[{mol}] nothing to write (no checkpoints found)\n")
        return

    df = (pd.DataFrame(rows)
            .sort_values(["method", "R_bohr"])
            .reset_index(drop=True))

    out_path = (Path(args.out).resolve() if args.out
                else mol_dir / f"quadrature_{mol}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, float_format="%.8f")

    print("=" * 96)
    print(df.to_string(index=False))
    print("=" * 96)
    print(f"[{mol}] saved → {out_path}\n")


for mol in selected:
    scan_molecule(mol)
