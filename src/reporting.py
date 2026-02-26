import pandas as pd


def balance_table(pairs: pd.DataFrame, vars: list[str]) -> pd.DataFrame:
    rows = []

    for v in vars:
        t = pd.to_numeric(pairs[f"t_{v}"], errors="coerce")
        c = pd.to_numeric(pairs[f"c_{v}"], errors="coerce")
        rows.append({
            "var": v,
            "treated_mean": t.mean(),
            "control_mean": c.mean(),
            "mean_diff": (t-c).mean(),
            "treated_sd": t.std(ddof=0),
            "control_sd": c.std(ddof=0),
        })

    return pd.DataFrame(rows)



def add_pair_deltas(pairs: pd.DataFrame, esg_col: str, div_col: str) -> pd.DataFrame:

    out = pairs.copy()
    out["delta_esg"] = pd.to_numeric(out[f"t_{esg_col}"], errors="coerce") - pd.to_numeric(out[f"c_{esg_col}"], errors="coerce")
    out["delta_div"] = pd.to_numeric(out[f"t_{div_col}"], errors="coerce") - pd.to_numeric(out[f"c_{div_col}"], errors="coerce")
    
    return out