import json
from typing import Dict, List, Literal, Optional
import numpy as np
import pandas as pd
from skbio.stats.distance import DistanceMatrix, mantel
from model.utils import save_json, load_json


def mantel_tests(save_json_path: Optional[str]):

    M_SED = transform_to_DistanceMatrix(
        model_type="SED", p="data/result/edit_distances(normalized).json"
    )
    M_TED = transform_to_DistanceMatrix(
        model_type="TED", p="data/result/edit_distances(normalized).json"
    )
    M_PPM_iso = transform_to_DistanceMatrix(
        model_type="PPM_iso",
        p="data/result/(ctx=False)/pairwise_similarity.json",
    )
    M_STITCH_iso = transform_to_DistanceMatrix(
        model_type="STITCH_iso",
        p="data/result/(ctx=False)/pairwise_similarity.json",
    )
    M_PPM_ctx = transform_to_DistanceMatrix(
        model_type="PPM_ctx", 
        p="data/result/(ctx=True)/pairwise_similarity.json"
    )
    M_STITCH_ctx = transform_to_DistanceMatrix(
        model_type="STITCH_ctx", 
        p="data/result/(ctx=True)/pairwise_similarity.json"
    )

    matrices = {
        "SED": M_SED,
        "TED": M_TED,
        "PPM_iso": M_PPM_iso,
        "STITCH_iso": M_STITCH_iso,
        "PPM_ctx": M_PPM_ctx,
        "STITCH_ctx": M_STITCH_ctx,
    }

    keys = list(matrices.keys())
    results = []

    # Mantel test for every unique pair of matrices
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            name1, name2 = keys[i], keys[j]
            m1, m2 = matrices[name1], matrices[name2]

            # Print mismatches (for diagnostics only)
            m1_ids = set(m1.ids)
            m2_ids = set(m2.ids)

            mismatched_in_m1 = m1_ids - m2_ids
            mismatched_in_m2 = m2_ids - m1_ids

            if mismatched_in_m1:
                print(
                    f"[{name1} vs {name2}] IDs in {name1} but not {name2}: {mismatched_in_m1}"
                )
            if mismatched_in_m2:
                print(
                    f"[{name1} vs {name2}] IDs in {name2} but not {name1}: {mismatched_in_m2}"
                )

            # Run Mantel test with 10,000 permutations
            # strict=False allows skbio to automatically subset to overlapping IDs
            corr, p_val, n = mantel(m1, m2, permutations=10000, strict=False)

            results.append(
                {
                    "Matrix 1": name1,
                    "Matrix 2": name2,
                    "corr": corr,
                    "p_value": p_val,
                    "n": n,
                }
            )

    if save_json_path:
        df_results = pd.DataFrame(results).to_dict(orient="records")
        save_json(fpath=save_json_path, res=df_results)
    else:
        print("\nTest Results:")
        print(pd.DataFrame(results))


def transform_to_DistanceMatrix(
    p: str,
    model_type: Literal["SED", "TED", "PPM_iso", "STITCH_iso", "PPM_ctx", "STITCH_ctx"],
) -> DistanceMatrix:

    df = pd.read_json(p)

    if model_type == "SED":
        D_field = f"{model_type}"

    elif model_type == "TED":
        D_field = f"{model_type}(zss)"

    elif model_type in ("PPM_iso", "STITCH_iso"):
        assert "(ctx=False)" in p, f"Expected '(ctx=False)' in path, but got: {p}"
        m = model_type.split("_")[0].lower()
        D_field = f"D_{m}"

    elif model_type in ("PPM_ctx", "STITCH_ctx"):
        assert "(ctx=True)" in p, f"Expected '(ctx=True)' in path, but got: {p}"
        m = model_type.split("_")[0].lower()
        D_field = f"D_{m}"

    # Normalize ID names by removing "(Valid)" prefix to match clean-labeled datasets
    df["nameX"] = df["nameX"].astype(str).str.replace(r"^\(Valid\)", "", regex=True)
    df["nameY"] = df["nameY"].astype(str).str.replace(r"^\(Valid\)", "", regex=True)

    ids = sorted(list(set(df["nameX"]).union(set(df["nameY"]))))
    dist_df = pd.DataFrame(0.0, index=ids, columns=ids)

    for _, row in df.iterrows():
        x, y, val = row["nameX"], row["nameY"], row[D_field]
        dist_df.loc[x, y] = val
        dist_df.loc[y, x] = val  # Distance matrices must be symmetric

    dm = DistanceMatrix(dist_df.values.copy(), ids)
    return dm
