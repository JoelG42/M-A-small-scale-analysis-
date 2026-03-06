import pandas as pd
from pathlib import Path
import statsmodels.formula.api as smf
from src.io import load_csv, save_csv
from src.features.regressions import valuation_regression, GB_regression
from src.config import MatchConfig


cfg = MatchConfig()
df = load_csv(Path(cfg.valuation_input_path))
valuation_results = valuation_regression(df)
GB_results = GB_regression(df)

save_csv(valuation_results, Path(cfg.output_dir) / "valuation_results.csv")
save_csv(GB_results, Path(cfg.output_dir) / "GB_results.csv")

