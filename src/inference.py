import numpy as np



def permutation_test_pair_diffs(deltas: np.ndarray, n_perm: int = 20000, seed: int = 1) -> dict:

    rng = np.random.default_rng(seed)
    deltas = np.asarray(deltas).astype(float)
    deltas = deltas[~np.isnan(deltas)]
    n = deltas.size

    if n == 0:
        raise ValueError("No valid deltas to test")


    obs = deltas.mean()
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))

    perm_means = (deltas * signs).mean(axis=1)
    P_two = (np.abs(perm_means) >= np.abs(obs)).mean()

    return {
        "n_pairs": int(n),
        "obs_mean": np.round(float(obs), 4),
        "p_value_two_sided": np.round(float(P_two), 4),
    }

