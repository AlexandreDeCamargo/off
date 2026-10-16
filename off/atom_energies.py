"""
Grid-integrated total energy vs EMA energy, side by side, for a set of atoms.

For each atom it:
  1. grid-integrates the LAST checkpoint via ``quadrature.grid_energy_from_checkpoint``
     ->  E_grid (= E_total; for a single atom E_NN = 0, so this is the electronic
     total energy).
  2. averages the last --window rows of training_metrics_ema.csv (E + CC) -> E_ema.
  3. writes both numbers side by side to a CSV (+ prints a table).

Because E_NN = 0 for an isolated atom, E_grid and E_ema are the *same* physical
quantity (total atomic energy); the only difference is grid quadrature vs the
Monte-Carlo EMA estimate during training.

Directory layout assumed (same as main.py):
    {results_root}/{atom}/{method}/bl_0.00/
        Checkpoints/checkpoint_*.eqx
        training_metrics_ema.csv
        job_params.json

Usage
-----
python atom_energies.py --method tf_w_lam0.2_none_lda_none_dopri8_atom_db_sched_MIX
python atom_energies.py --method <tag> --atoms B Be C F H He Li N Ne O \
    --window 1000 --recompute --results_root /path/to/Results
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
from pathlib import Path

import pandas as pd

from quadrature import grid_energy_from_checkpoint

# atomic number == electron count for a neutral atom
Z_TABLE = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
           "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--method", type=str, required=True,
                    help="Method directory name, e.g. "
                         "tf_w_lam0.2_none_lda_none_dopri8_atom_db_sched_MIX")
parser.add_argument("--atoms", type=str, nargs="+",
                    default=["B", "Be", "C", "F", "H", "He", "Li", "N", "Ne", "O"],
                    help="Atoms to process (default: the full set).")
parser.add_argument("--results_root", type=str, default="Results",
                    help="Root dir holding {atom}/{method}/bl_0.00 (default: Results)")
parser.add_argument("--window", type=int, default=1000,
                    help="Average the last N rows of training_metrics_ema.csv (default: 1000)")
parser.add_argument("--bs", type=int, default=256, help="Grid chunk size")
parser.add_argument("--grid_level", type=int, default=3, help="PySCF grid level")
parser.add_argument("--recompute", action="store_true",
                    help="Re-run grid integration even if energy_summary.json is cached")
parser.add_argument("--out", type=str, default="atom_energies.csv",
                    help="Output CSV path (default: atom_energies.csv)")
args = parser.parse_args()


def read_last_ema(bl_dir: Path, window: int):
    """Mean of the last `window` rows of training_metrics_ema.csv.
    E = E + CC  (no nuclear repulsion; for an atom this is the total energy)."""
    csv = bl_dir / "training_metrics_ema.csv"
    if not csv.exists():
        return None, None
    try:
        df = pd.read_csv(csv)
    except pd.errors.EmptyDataError:
        return None, None
    if df.empty:
        return None, None
    tail = df.tail(window)
    E = float(tail["E"].mean())
    if "CC" in tail.columns:
        E += float(tail["CC"].mean())
    epoch = int(df.iloc[-1]["epoch"])
    return E, epoch


# ── main loop ─────────────────────────────────────────────────────────────────
root = Path(args.results_root).resolve()
print(f"Results root : {root}")
print(f"Method       : {args.method}")
print(f"EMA window   : last {args.window} rows\n")

rows = []
for atom in args.atoms:
    atom_dir = root / atom / args.method / "bl_0.00"
    print(f"[{atom}]  {atom_dir}")

    if not (atom_dir / "job_params.json").exists():
        print("    SKIP — no job_params.json (directory missing?)\n")
        continue

    # grid integration of the last checkpoint (via the OFF quadrature module)
    try:
        data = grid_energy_from_checkpoint(
            atom_dir, grid_level=args.grid_level, chunk=args.bs, recompute=args.recompute)
        E_grid     = data["E_total"]
        grid_epoch = data["epoch"]
        Ne_int     = data["Ne_integral"]
    except Exception as e:
        print(f"    grid integration FAILED: {e}")
        E_grid = grid_epoch = Ne_int = None

    # EMA mean of last `window` rows
    E_ema, ema_epoch = read_last_ema(atom_dir, args.window)
    if E_ema is None:
        print("    no training_metrics_ema.csv")

    diff = (E_grid - E_ema) if (E_grid is not None and E_ema is not None) else None
    rows.append({
        "atom":       atom,
        "Ne":         Z_TABLE.get(atom),
        "E_grid_Ha":  E_grid,
        "grid_epoch": grid_epoch,
        "E_ema_Ha":   E_ema,
        "ema_epoch":  ema_epoch,
        "diff_Ha":    diff,
        "Ne_int":     Ne_int,
    })
    if E_grid is not None and E_ema is not None:
        print(f"    E_grid={E_grid:+.6f}  E_ema={E_ema:+.6f}  Δ={diff:+.6f} Ha")
    print()

if not rows:
    raise RuntimeError("No atoms processed — check --results_root / --method.")

df = pd.DataFrame(rows)

# ── print + save ──────────────────────────────────────────────────────────────
print("=" * 80)
print(f"{'atom':>4}  {'Ne':>3}  {'E_grid [Ha]':>15}  {'E_ema [Ha]':>15}  "
      f"{'Δ(grid-ema)':>13}  {'∫ρ':>8}")
print("-" * 80)
for _, r in df.iterrows():
    eg = f"{r['E_grid_Ha']:+15.6f}" if pd.notna(r['E_grid_Ha']) else f"{'—':>15}"
    em = f"{r['E_ema_Ha']:+15.6f}"  if pd.notna(r['E_ema_Ha'])  else f"{'—':>15}"
    dd = f"{r['diff_Ha']:+13.6f}"   if pd.notna(r['diff_Ha'])   else f"{'—':>13}"
    ni = f"{r['Ne_int']:8.4f}"      if pd.notna(r['Ne_int'])    else f"{'—':>8}"
    ne = f"{int(r['Ne'])}"          if pd.notna(r['Ne'])        else "?"
    print(f"{r['atom']:>4}  {ne:>3}  {eg}  {em}  {dd}  {ni}")
print("=" * 80)

out_path = Path(args.out).resolve()
df.to_csv(out_path, index=False, float_format="%.8f")
print(f"\nSaved → {out_path}")
