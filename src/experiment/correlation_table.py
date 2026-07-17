from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanrho

from model.utils import load_json, save_json


def remove_valid_in_titles(data: List[Dict[str, Any]]):
    results = [
        {
            **p,
            "nameX": p["nameX"].replace("(Valid)", "").strip(),
            "nameY": p["nameY"].replace("(Valid)", "").strip(),
        }
        for p in data
    ]
    return results


def generate_corr_table(
    results_noctx_path: str,
    results_ctx_path: str,
    results_other_path: str,
    output_dir: str,
    corr_fun: callable,
) -> None:
    """
    Generate correltation matrix LaTeX table.
    """
    latex_strs = {
        "D_stitch_noctx": "$D^{\\textsc{Stitch}}$",
        "D_ppm_noctx": "$D^{\\textsc{PPM}}$",
        "D_stitch_ctx": "$D^{\\textsc{Stitch}}_\\mathcal{T}$",
        "D_ppm_ctx": "$D^{\\textsc{PPM}}_\\mathcal{T}$",
    }

    data_noctx = remove_valid_in_titles(load_json(results_noctx_path))
    data_ctx = remove_valid_in_titles(load_json(results_ctx_path))
    data_other = remove_valid_in_titles(load_json(results_other_path))

    df_noctx = pd.DataFrame(data_noctx)[["nameX", "nameY", "D_stitch", "D_ppm"]]
    df_ctx = pd.DataFrame(data_ctx)[["nameX", "nameY", "D_stitch", "D_ppm"]]
    df_other = pd.DataFrame(data_other)
    df = pd.merge(df_noctx, df_ctx, on=["nameX", "nameY"], suffixes=("_noctx", "_ctx"))
    df = pd.merge(df, df_other, on=["nameX", "nameY"])

    cols = [
        "D_stitch_noctx",
        "D_ppm_noctx",
        "D_stitch_ctx",
        "D_ppm_ctx",
        "SED",
        "TED(zss)",
    ]
    output_df = pd.DataFrame({"": cols})
    for i, cx in enumerate(cols[:-1]):
        col_values = [""] * i
        for cy in cols[i:]:
            if cx == cy:
                col_values.append("--")
            else:
                col_values.append(f"{corr_fun(df[cx], df[cy])[0]:.{3}f}")
        output_df[cx] = col_values

    output_df.to_csv(output_dir + "correlation_table.csv", index=False)

    output_df.to_latex(output_dir + "correlation_table.tex", index=False)
