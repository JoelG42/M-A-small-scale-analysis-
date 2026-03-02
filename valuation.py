import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf


def stars(p):
    if p is None or pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    elif p < 0.05:
        return "**"
    elif p < 0.10:
        return "*"
    else:
        return ""


#Load matched pairs 
pairs = pd.read_csv("outputs/matched_pairs.csv")


#Build regression dataframe
reg_df = pd.concat([
    pairs.filter(regex="^t_").rename(columns=lambda x: x[2:]),
    pairs.filter(regex="^c_").rename(columns=lambda x: x[2:]),
])

reg_df["pair_id"] = list(range(len(pairs))) * 2

valuation_vars = ["log_deal_value", "premium_1d", "premium_1w", "premium_1m", "deal_to_ebitda", "deal_to_net_assets"]
divergence_vars = ["log_SD_ESG", "log_SD_E", "log_SD_S", "log_SD_G", "SD_ESG", "SD_E", "SD_S", "SD_G", "CV_ESG", "CV_E", "CV_S", "CV_G"]
pillars = ["ESG", "E", "S", "G"]


results = []

for v in valuation_vars:
    for p in pillars:

        level_var = f"mean_{p}"
        div_var = f"CV_{p}"
        needed = [v, level_var, div_var, "has_green_bond", "pair_id"]
        df_sub = reg_df[needed].dropna(subset=needed)

        if len(df_sub) < 10:
            continue

        formula = f"{v} ~ has_green_bond + {level_var} + {div_var} + {div_var}:has_green_bond"

        model = smf.ols(
            formula,
            data=df_sub,
        ).fit(cov_type="cluster",
         cov_kwds={"groups": df_sub["pair_id"]})

        results.append({
            "Valuation": v,
            "Pillar": p,
            "Divergence": div_var,
            "beta_level": model.params.get(level_var),
            "p_level": model.pvalues.get(level_var),
            "sig_level": stars(model.pvalues.get(level_var)),
            "beta_div": model.params.get(div_var),
            "beta_interaction": model.params.get(f"{div_var}:has_green_bond"),
            "p_div": model.pvalues.get(div_var),
            "sig_div": stars(model.pvalues.get(div_var)),
            "p_interaction": model.pvalues.get(f"{div_var}:has_green_bond"),
            "sig_interaction": stars(model.pvalues.get(f"{div_var}:has_green_bond")),
            "n": model.nobs,
         })


results_df = pd.DataFrame(results)
results_df.to_csv("outputs/valuation_results.csv", index=False)