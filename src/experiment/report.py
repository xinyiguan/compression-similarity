from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanrho, zscore, linregress

from model.utils import load_json, save_json


def remove_valid_in_titles(data:List[Dict[str, Any]]):
    results = [
        {
            **p,
            "nameX": p["nameX"].replace("(Valid)", "").strip(),
            "nameY": p["nameY"].replace("(Valid)", "").strip(),
        }
        for p in data
    ]
    return results


def corr_rank_report(
    res_file: str,
    models: Tuple[str, str],
    save_report_fpath: Optional[str],
    show_results: bool,
):
    _data = load_json(res_file)
    data = remove_valid_in_titles(_data)
    m1 = models[0]
    m2 = models[1]

    print(f"Comparing model {m1} and {m2} \n")

    df = pd.DataFrame(data)
    df = df[["nameX", "nameY", f"D_{m1}", f"D_{m2}"]].copy()
    df["pair"] = df["nameX"] + "  |  " + df["nameY"]

    print(f"== correlation ==")
    pearson_r, pearson_p = pearsonr(df[f"D_{m1}"], df[f"D_{m2}"])
    spearman = spearmanrho(df[f"D_{m1}"], df[f"D_{m2}"])
    spearman_rho, spearman_p = spearman.statistic, spearman.pvalue
    kendall = kendalltau(df[f"D_{m1}"], df[f"D_{m2}"])
    kendall_tau, kendall_p = kendall.statistic, kendall.pvalue

    corr_report = {
        "Pearson r": f"{pearson_r:.4f} (p={pearson_p:.4e})",
        "Spearman rho": f"{spearman_rho:.4f} (p={spearman_p:.4e})",
        "Kendall tau": f"{kendall_tau:.4f} (p={kendall_p:.4e})",
    }

    print(f' == regression ==')
    linRes = linregress(x=df[f"D_{m2}"], y=df[f"D_{m1}"], nan_policy="raise")

    regression_report = {
        "slope": linRes.slope,
        "intercept": linRes.intercept,
        "rval": linRes.rvalue,
        "R^2": (linRes.rvalue) * (linRes.rvalue),
        "pval": linRes.pvalue,
        "stderr": linRes.stderr,
    }

    # zscored distance =====
    print(f'Computing the z-scored distance')
    df[f"D_{m1}(z-score)"] = zscore(df[f"D_{m1}"])
    df[f"D_{m2}(z-score)"] = zscore(df[f"D_{m2}"])

    # distance diff (after z-scored) ====
    df["dist_diff"] = df[f"D_{m1}(z-score)"] - df[f"D_{m2}(z-score)"]
    df["abs_dist_diff"] = df["dist_diff"].abs()

    # largest rank disagreement
    top_k_disagree = df.sort_values("dist_diff", ascending=False)
    top_k_agree = df.sort_values("dist_diff", ascending=True)
    bottom_k_disagree = df.sort_values("abs_dist_diff", ascending=True)

    report = {
        "models in comparison": f"{m1} and {m2}",
        "correlation results": corr_report,
        "regression results": regression_report,
        "Top k disagree": top_k_disagree.to_dict(orient="records"),
        f"To k agree": top_k_agree.to_dict(orient="records"),
        f"Bottom k disagree": bottom_k_disagree.to_dict(orient="records"),
    }

    if save_report_fpath:
        save_json(save_report_fpath, report)

    if show_results:
        # showing results
        show(
            df_subset=top_k_disagree,
            models=(m1, m2),
            title="Top k disagree",
            k=10,
        )

        show(
            df_subset=top_k_agree,
            models=(m1, m2),
            title=f"To k agree",
            k=10,
        )

        show(
            df_subset=bottom_k_disagree,
            models=(m1, m2),
            title=f"Bottom k disagree",
            k=10,
        )


def show(df_subset:pd.DataFrame, models: Tuple[str, str], title:str, k:int):
    m1 = models[0]
    m2 = models[1]

    print(f"\n=== {title} (top {k}) ===")
    cols = [
        "pair",
        # f"D_{m1}",
        # f"D_{m2}",
        f"D_{m1}(z-score)",
        f"D_{m2}(z-score)",
        "dist_diff"
        # "abs_dist_diff",
    ]
    res=df_subset[cols].head(k)
    print(res.to_string(index=False))
    return res
