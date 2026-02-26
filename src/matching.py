from __future__ import annotations
import numpy as np
import pandas as pd 
from sklearn.neighbors import NearestNeighbors


def _standardize(train: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    mu = train.mean()
    sd = train.std(ddof=0).replace(0, 1.0)
    return (X - mu) / sd



def match_pairs(
    df: pd.DataFrame,
    treat_col: str,
    year_col: str,
    exact_cols: List[str],
    distance_cols: List[str],
    calipers: Dict[str, float] | None = None,
    k_neighbors: int = 1,
    replace: bool = False,
) -> pd.DataFrame:


    calipers = calipers or {}

    needed = [treat_col, year_col] + exact_cols + distance_cols
    d = df.dropna(subset=needed).copy()


    treated = d[d[treat_col]== 1]
    controls = d[d[treat_col]== 0]

    if treated.empty:
        raise ValueError("No treated units found in the dataframe")

    if controls.empty:
        raise ValueError("No control units found in the dataframe")


    ctrl_train = controls[distance_cols].astype(float)
    treated_Z = _standardize(ctrl_train, treated[distance_cols].astype(float))
    controls_Z = _standardize(ctrl_train, controls[distance_cols].astype(float))


    used_controls: set[int] = set()
    matches: list[tuple[int, int, float]] = []


    for t_idx, t_row in treated.iterrows():
        mask = (controls[year_col] == t_row[year_col])
        for c in exact_cols:
            mask &= (controls[c] == t_row[c])


        pool = controls[mask]
        if not replace:
            pool = pool[~pool.index.isin(used_controls)]
        if pool.empty:
            continue

        ok = pd.Series(True, index=pool.index)
        for col, width in calipers.items():
            ok &= (pool[col] - t_row[col]).abs() <= width
        pool = pool[ok]
        if pool.empty:
            continue


        poolZ = controls_Z.loc[pool.index]
        tvec = treated_Z.loc[t_idx]

        nn = NearestNeighbors(n_neighbors=min(k_neighbors, len(poolZ)), metric="euclidean")
        nn.fit(poolZ.to_numpy())

        dist, ind = nn.kneighbors(tvec.to_numpy().reshape(1, -1))


        for j in range(ind.shape[1]):
            c_idx = poolZ.index[ind[0, j]]
            matches.append((t_idx, c_idx, float(dist[0, j])))
            if not replace:
                used_controls.add(c_idx)


        
    pairs = pd.DataFrame(matches, columns=["treated_idx", "control_idx", "distance"])
    if pairs.empty:
        raise ValueError("No matches found under your exact constraints + calipers")

    
    t_side = d.loc[pairs["treated_idx"]].add_prefix("t_").reset_index(drop=True)
    c_side = d.loc[pairs["control_idx"]].add_prefix("c_").reset_index(drop=True)

    out = pd.concat([pairs.reset_index(drop=True), t_side, c_side], axis=1)
    return out