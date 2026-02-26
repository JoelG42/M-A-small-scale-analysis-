from pathlib import Path
import pandas as pd

from src.io import load_excel, ensure_columns, save_csv
from src.matching import match_pairs
from src.inference import permutation_test_pair_diffs
from src.reporting import balance_table, add_pair_deltas
from src.config import MatchConfig


def run_all(cfg: MatchConfig) -> None:
    df = load_excel(cfg.input_path) 


    required = (
        [cfg.treat_col, cfg.year_col]
        + cfg.exact_cols
        + cfg.distance_cols
        + [cfg.esg_col, cfg.div_col]
    )

    ensure_columns(df, required)


    pairs = match_pairs(
        df=df,
        treat_col=cfg.treat_col,
        year_col=cfg.year_col,
        exact_cols=cfg.exact_cols,
        distance_cols=cfg.distance_cols,
        calipers=cfg.calipers,
        k_neighbors=cfg.k_neighbors,
        replace=cfg.replace,
    )

    bal = balance_table(pairs, cfg.distance_cols)
    pairs2 = add_pair_deltas(pairs, cfg.esg_col, cfg.div_col)

    res_esg = permutation_test_pair_diffs(pairs2["delta_esg"].values, n_perm=cfg.n_perm, seed=cfg.seed)
    res_div = permutation_test_pair_diffs(pairs2["delta_div"].values, n_perm=cfg.n_perm, seed=cfg.seed + 1)


    outdir = Path(cfg.output_dir)
    save_csv(bal, outdir / "balance_table.csv")
    save_csv(pairs2, outdir / "matched_pairs.csv")
    save_csv(pd.DataFrame([{"outcome": "esg", **res_esg}, {"outcome": "div", **res_div}]), outdir / "permutation_tests.csv")

    return {"pairs": pairs2, "balance": bal, "esg": res_esg, "div": res_div}