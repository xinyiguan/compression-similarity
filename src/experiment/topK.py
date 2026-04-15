
from typing import Literal

from src.model.utils import load_json, save_json


def get_topK_compressed(res_path: str, k: int, save_fpath: str):
    """
    sort the C value (compressed encoding size)
    """
    
    raise NotImplementedError

def get_topK_similar_pair(res_path: str, k: int, model: Literal["ppm", "stitch"], save_fpath: str):
    """
    sort the D values in selected model from small to large, 
    return the top K smallest with their d values 
    """
    d_res = load_json(fpath=res_path)
    key = f"D_{model}"

    entries = [e for e in d_res if key in e] 
    
    entries_sorted = sorted(entries, key=lambda e: e.get(key, float('inf'))) # sort ascending (smaller D -> more similar)

    topk = entries_sorted[:k]

    # build concise result
    result = [{
        "nameX": e.get("nameX"),
        "nameY": e.get("nameY"),
        key: e.get(key)
    } for e in topk]

    if save_fpath:
        save_json(fpath=save_fpath, res=result)

    return result
    
    