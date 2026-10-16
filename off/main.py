import argparse
import json
from fractions import Fraction
from pathlib import Path
from .train.loop import training
from .config._config import Config


def _lam(value: str) -> float:
    """Accept λ as a fraction ('1/9', '1/5') or plain float ('0.2', '2.0')."""
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        raise argparse.ArgumentTypeError(
            f"Invalid λ value '{value}'. Use a fraction (1/9, 1/5) or float (0.111, 2.0)."
        )

SINGLE_ATOMS = {'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne'}

def _method_tag(args) -> str:
    """Encode functional/solver choices into a compact directory name."""
    kin_tag = args.kin
    if args.kin in ('w', 'tf_w'):
        kin_tag += f"_lam{args.lam:.6g}"

    tag = f"{kin_tag}_{args.cc}_{args.x}_{args.c}_{args.solver}_{args.prior}"
    if args.sched.lower() not in ['c', 'const']:
        tag += f"_sched_{args.sched}"
    if args.hart.lower() != 'coulomb':
        tag += f"_hart_{args.hart}"
    return tag.lower()


def setup_directories(args):
    """Create and return directory paths for results, checkpoints, and figures.

    Layout:  Results/{mol}/{method}/bl_{bond_length}/
      - Single atoms always use bl_0.00 (bond length has no meaning).
      - Diatomics/polyatomics use the supplied --bond_length value.
    This makes bond-length scans trivial:
      glob('Results/H2/{method}/bl_*/')
    """
    bl = 0.0 if args.mol_name in SINGLE_ATOMS else args.bond_length
    results_dir = f"Results/{args.mol_name}/{_method_tag(args)}/bl_{bl:.2f}"
    ckpt_dir    = f"{results_dir}/Checkpoints"

    for directory in [results_dir, ckpt_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)

    return results_dir, ckpt_dir


def save_job_params(results_dir, args):
    """Save training parameters to JSON file."""
    job_params = {
        'model': 'cnf',
        'mol_name': args.mol_name,
        'bond_length': args.bond_length,
        'epochs': args.epochs,
        'batch_size': args.bs,
        'hidden_layer': args.hl,
        'lr': args.lr,
        'kinetic': args.kin,
        'lam': args.lam,
        'external': args.nuc,
        'hartree': args.hart,
        'exchange': args.x,
        'correlation': args.c,
        'core_correction': args.cc,
        'scheduler': args.sched,
        'solver': args.solver,
        'prior': args.prior,
    }

    with open(f"{results_dir}/job_params.json", "w") as outfile:
        json.dump(job_params, outfile, indent=4)

    return job_params


def main():
    parser = argparse.ArgumentParser()
    # Model parameters
    parser.add_argument("--mol_name", type=str, default='H2',
                        help="Molecule name")
    parser.add_argument("--bond_length", type=float, default=4.4,
                        help="Bond length for the molecule (Bohr)")
    parser.add_argument("--epochs", type=int, default=500,
                        help="Number of training epochs")
    parser.add_argument("--bs", type=int, default=512,
                        help="Batch size")
    parser.add_argument("--hl", type=int, default=64,
                        help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--prior", type=str, default='promolecular',
                    choices=['promolecular', 'atom_db'],
                    help="Prior distribution type")

    # Functionals
    parser.add_argument("--kin", type=str, default='tf_w',
                        choices=['tf', 'w', 'tf_w'],
                        help="Kinetic energy functional")
    parser.add_argument("--lam", type=_lam, default=1/5,
                        help="Weizsäcker prefactor λ in TF-λW: fraction or float ")
    parser.add_argument("--nuc", type=str, default='np',
                        help="Nuclear potential functional")
    parser.add_argument("--hart", type=str, default='coulomb',
                        help="Hartree energy functional")
    parser.add_argument("--x", type=str, default='lda',
                        choices=['lda', 'b88_x', 'lda_b88_x'],
                        help="Exchange energy functional")
    parser.add_argument("--c", type=str, default='none',
                        choices=['vwn_c', 'pw92_c', 'none'],
                        help="Correlation energy functional")
    parser.add_argument("--cc", type=str, default='none',
                        choices=['kato', 'hutcheon', 'none'],
                        help="Core correction functional")

    # Training settings
    parser.add_argument("--sched", type=str, default='mix',
                        help="Learning rate scheduler type")
    parser.add_argument("--solver", type=str, default='dopri8',
                        choices=['dopri5', 'tsit5', 'dopri8'],
                        help="ODE solver")
    parser.add_argument("--ckpt_freq", type=int, default=15,
                        help="Checkpoint saving frequency (epochs)")

    args = parser.parse_args()

    Config.from_args(args)

    # Setup directories
    results_dir, ckpt_dir = setup_directories(args)
    Config.set_directories(results_dir, ckpt_dir)

    # Save parameters
    job_params = save_job_params(results_dir, args)
    print(f"Starting training with parameters:")
    print(json.dumps(job_params, indent=2))
    print(f"\nResults will be saved to: {results_dir}")

    # Run training
    shared = dict(
        mol_name=args.mol_name,
        bond_length=args.bond_length,
        tw_kin=args.kin,
        lam=args.lam,
        n_pot=args.nuc,
        h_pot=args.hart,
        x_pot=args.x,
        c_pot=args.c,
        cc_pot=args.cc,
        batch_size=args.bs,
        hidden_layer=args.hl,
        epochs=args.epochs,
        lr=args.lr,
        scheduler_type=args.sched,
        prior_type=args.prior,
        checkpoint_dir=ckpt_dir,
        checkpoint_freq=args.ckpt_freq,
    )

    model, df, df_ema = training(**shared, solver_type=args.solver)

    print(f"\nTraining complete!")
    print(f"Results saved to: {results_dir}")
    print(f"Final energy (EMA): {df_ema['E'].iloc[-1]:.6f}")


if __name__ == "__main__":
    main()
