import re
from typing import Any, Dict, List, Literal, Optional

import pandas as pd

from model.utils import load_json, save_json


def get_topK_similar_pair(
    res_path: str, k: int, model: Literal["ppm", "stitch"]
):
    """
    sort the D values in selected model from small to large,
    return the top K smallest with their d values
    """
    d_res = load_json(fpath=res_path)
    key = f"D_{model}"

    entries_set = {
        ",".join(sorted([e["nameX"], e["nameY"]])): e[key] for e in d_res if key in e
        and e["nameX"] != e["nameY"]
    }
    entries = [
        {"nameX": k.split(",")[0], "nameY": k.split(",")[1], key:v}
        for k, v in entries_set.items()
    ]

    entries_sorted = sorted(
        entries, key=lambda e: e.get(key, float("inf"))
    )  # sort ascending (smaller D -> more similar)

    topk = entries_sorted[:k]

    # build concise result
    result = [
        {"nameX": e.get("nameX"), "nameY": e.get("nameY"), key: e.get(key)}
        for e in topk
    ]

    return result



def pprint_topK_table(
    topK_res: List[Dict[str, Any]], save_fpath: Optional[str] = None
) -> str:

    if not topK_res:
        raise ValueError("Input list is empty.")

    # detect distance key
    d_key = next((k for k in topK_res[0].keys() if k.startswith("D_")), None)
    if d_key is None:
        raise ValueError("No distance key found.")

    model = d_key.replace("D_", "")

    # --- escape ---
    def latex_escape(text: str) -> str:
        text = str(text)
        replacements = {
            "&": r"\&",
            "%": r"\%",
            "$": r"\$",
            "#": r"\#",
            "_": r"\_",
            "{": r"\{",
            "}": r"\}",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        return text

    df = pd.DataFrame(topK_res)

    df[d_key] = df[d_key].map(lambda x: f"{x:.4f}")
    df["nameX"] = df["nameX"].map(latex_escape)
    df["nameY"] = df["nameY"].map(latex_escape)

    df = df.rename(columns={"nameX": "Piece X", "nameY": "Piece Y", d_key: "NCD"})

    df.insert(0, "Rank", range(1, len(df) + 1))

    latex_str = df.to_latex(
        index=False,
        escape=False,
        column_format="rlll",
        caption=f"Top {len(df)} most similar pieces ({model})",
        label=f"tab:topk_{model}",
    )

    if save_fpath:
        with open(save_fpath, "w") as f:
            f.write(latex_str)

    return latex_str
