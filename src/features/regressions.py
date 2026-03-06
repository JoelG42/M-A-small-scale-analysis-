import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from src.config import MatchConfig


cfg = MatchConfig()

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






def valuation_regression(df: pd.DataFrame) -> pd.DataFrame:
    reg_df = pd.concat([df.filter(regex="^t_").rename(columns=lambda x: x[2:]), df.filter(regex="^c_").rename(columns=lambda x: x[2:])])

    results = []
    for v in cfg.valuation_vars:
        for p in cfg.pillars:
            level_var = f"mean_{p}"
            div_var = f"CV_{p}"
            gb = cfg.treat_col
            needed = [v, level_var, div_var, gb]
            df_sub = reg_df[needed].dropna(subset=needed)
            if len(df_sub) < 10:
                continue
            formula = f"{v} ~ {gb} + {level_var} + {div_var} + {div_var}:{gb}"

            model = smf.ols(formula, data=df_sub).fit()
            results.append({
                "Valuation": v,
                "Pillar": p,
                "Divergence": div_var,
                "beta_level": model.params.get(level_var),
                "p_level": model.pvalues.get(level_var),
                "sig_level": stars(model.pvalues.get(level_var)),
                "beta_div": model.params.get(div_var),
                "p_div": model.pvalues.get(div_var),
                "sig_div": stars(model.pvalues.get(div_var)),
                "beta_interaction": model.params.get(f"{div_var}:{gb}"),
                "p_interaction": model.pvalues.get(f"{div_var}:{gb}"),
                "sig_interaction": stars(model.pvalues.get(f"{div_var}:{gb}")),
                "n": model.nobs,
            })
    results_df = pd.DataFrame(results)
    return results_df



def GB_regression(df: pd.DataFrame) -> pd.DataFrame:
    reg_df = pd.concat([df.filter(regex="^t_").rename(columns=lambda x: x[2:]), df.filter(regex="^c_").rename(columns=lambda x: x[2:])])
    results = []
    for m in cfg.div_measures:
        for p in cfg.pillars:
            level_var = f"mean_{p}"
            div_var = f"{m}{p}"
            needed = [cfg.treat_col, level_var, div_var]
            df_sub = reg_df[needed].dropna(subset=needed)
            if len(df_sub) < 10:
                continue
            formula = f"{cfg.treat_col} ~ {level_var} + {div_var}"
            model = smf.ols(formula, data=df_sub).fit()
            results.append({
                "Pillar": p,
                "Divergence measure": m,
                "beta_level": model.params.get(level_var).round(4),
                "std_level": model.bse.get(level_var).round(4),
                "p_level": model.pvalues.get(level_var).round(4),
                "sig_level": stars(model.pvalues.get(level_var)),
                "beta_div": model.params.get(div_var).round(4),
                "std_div": model.bse.get(div_var).round(4),
                "p_div": model.pvalues.get(div_var).round(4),
                "sig_div": stars(model.pvalues.get(div_var)),
                "adj_r2": round(model.rsquared_adj, 4),
                "f_stat": round(model.fvalue, 4),
                "n": model.nobs,
            })
    results_df = pd.DataFrame(results)
    return results_df


