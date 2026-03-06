from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Literal


@dataclass(frozen=True)
class MatchConfig:

        input_path: str = "data/data_for_regression_GB.xlsx"
        valuation_input_path: str = "outputs/matched_pairs.csv"
        output_dir: str = "output"

        treat_col: str = "has_green_bond"
        year_col: str = "deal_year"

        exact_cols: List[str] = None
        distance_cols: List[str] = None
        calipers: Optional[Dict[str, float]] = None
        k_neighbors: int = 1
        replace: bool = False

        match_method: Literal["exact", "knn"] = "knn"

        n_perm: int = 20000
        seed: int = 1

        esg_col: str = ""
        div_col: str = ""

        valuation_vars = ["log_deal_value", "premium_1d", "premium_1w", "premium_1m", "deal_to_ebitda", "deal_to_net_assets"]
        pillars = ["ESG", "E", "S", "G"]
        div_measures = ["SD_", "log_SD_", "CV_"]


        def __post_init__(self):
                object.__setattr__(self, "exact_cols", self.exact_cols or ["cross_nation", "cross_industry", "IG"])
                object.__setattr__(self, "distance_cols", self.distance_cols or ["target_total_assets"])
                object.__setattr__(self, "calipers", self.calipers or {"target_total_assets": None})














TREAT_COL = "has_green_bond"
YEAR_COL = "deal_year"
BIN_EXACT = ["cross_nation", "cross_industry", "IG"]
SIZE_COL = "relative_total_assets"

DEALVALUE_COL = "log_deal_value"

ESG_COL = ["mean_ESG", "mean_E", "mean_S", "mean_G"]
DIV_COL = ["log_SD_ESG", "log_SD_E", "log_SD_S", "log_SD_G", "CV_ESG", "CV_E", "CV_S", "CV_G"]