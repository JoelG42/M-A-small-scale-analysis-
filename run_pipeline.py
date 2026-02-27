from pathlib import Path
import argparse

from src.config import MatchConfig
from src.runner import run_all

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str, help="Path to Excel file")
    p.add_argument("--outdir", default="outputs", type=str)
    p.add_argument("--esg_col", required=True, type=str, help="Column name of ESG score outcome")
    p.add_argument("--div_col", required=True, type=str, help="Column name of ESG divergence outcome")

    # optional overrides
    p.add_argument("--k", default=1, type=int, help="Neighbors per treated (1 recommended)")
    p.add_argument("--replace", action="store_true", help="Allow control reuse")
    p.add_argument("--match_method", choices=["exact", "knn"], default="exact", help="Matching method to use")
    p.add_argument("--n_perm", default=20000, type=int)
    p.add_argument("--seed", default=1, type=int)
    return p.parse_args()

def main():
    args = parse_args()

    cfg = MatchConfig(
        input_path=Path(args.input),
        output_dir=Path(args.outdir),
        esg_col=args.esg_col,
        div_col=args.div_col,
        k_neighbors=args.k,
        replace=args.replace,
        match_method=args.match_method,
        n_perm=args.n_perm,
        seed=args.seed,
    )

    result = run_all(cfg)

    # Minimal console output
    print("Matched pairs saved to:", Path(args.outdir) / "matched_pairs.csv")
    print("Balance table saved to:", Path(args.outdir) / "balance_table.csv")
    print("Permutation results saved to:", Path(args.outdir) / "permutation_results.csv")
    print("Permutation ESG:", result["esg"])
    print("Permutation DIV:", result["div"])

if __name__ == "__main__":
    main()
