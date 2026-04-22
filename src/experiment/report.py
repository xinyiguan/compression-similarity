from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanrho, zscore

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

    # zscored distance =====
    print(f'Computing the z-scored distance')
    df[f"D_{m1}(z-score)"] = zscore(df[f"D_{m1}"])
    df[f"D_{m2}(z-score)"] = zscore(df[f"D_{m2}"])

    # distance diff (after z-scored) ====
    df["dist_diff"] = df[f"D_{m1}(z-score)"] - df[f"D_{m2}(z-score)"]
    df["abs_dist_diff"] = df["dist_diff"].abs()

    # largest rank disagreement
    top_rank_disagree = df.sort_values("abs_dist_diff", ascending=False)

    # m1 says similar but m2 says dissimilar
    low_m1 = df[f"D_{m1}(z-score)"].quantile(0.25)  # bottom 25% = "similar"
    high_m2 = df[f"D_{m2}(z-score)"].quantile(0.75)  # top 25% = "dissimilar"

    m1_sim_m2_diff = df[
        (df[f"D_{m1}(z-score)"] <= low_m1) & (df[f"D_{m2}(z-score)"] >= high_m2)
    ].sort_values("abs_dist_diff", ascending=False)

    # m2 says similar but m1 says dissimilar
    low_m2 = df[f"D_{m2}(z-score)"].quantile(0.25) 
    high_m1 = df[f"D_{m1}(z-score)"].quantile(0.75)

    m2_sim_m1_diff = df[
        (df[f"D_{m2}(z-score)"] <= low_m2) & (df[f"D_{m1}(z-score)"] >= high_m1)
    ].sort_values("abs_dist_diff", ascending=False)

    report = {
        "models in comparison": f"{m1} and {m2}", 
        "correlation results": corr_report,
        "Largest distance disagreement": top_rank_disagree.to_dict(orient="records"),
        f" {m1} similar / {m2} dissimilar": m1_sim_m2_diff.to_dict(orient="records"),
        f" {m2} similar / {m1} dissimilar": m2_sim_m1_diff.to_dict(orient="records")
    }

    if save_report_fpath:
        save_json(save_report_fpath, report)

    if show_results:
        # showing results
        show(
            df_subset=top_rank_disagree,
            models=(m1, m2),
            title="Largest disagreement",
            k=10,
        )

        show(
            df_subset=m1_sim_m2_diff,
            models=(m1, m2),
            title=f" {m1} similar / {m2} dissimilar",
            k=10,
        )

        show(
            df_subset=m2_sim_m1_diff,
            models=(m1, m2),
            title=f" {m2} similar / {m1} dissimilar",
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
    res=df_subset[cols].head(k).to_string(index=False)
    print(res)
    return res
