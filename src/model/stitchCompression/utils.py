from collections import Counter
import json
import os
from typing import Dict, List, Tuple

from stitch_core import Abstraction


def extract_abstraction_rhs(abstraction_str: str):
    """
    Extract the right-hand side of 'fn_i(...) := ...'
    """
    return abstraction_str.split(" := ", 1)[1].strip()

def extract_piece_abstractions(piece: List[str]) -> Tuple[list,int]:
    """
    piece: list of abstraction strings for one piece
    
    Returns:
        sets: abstractions (for Jaccard)
        counts: (for weighted variants later)
    """
    rhs_list = [extract_abstraction_rhs(a) for a in piece]
    abstraction_set = set(rhs_list)
    abstraction_counts = Counter(rhs_list)
    
    return abstraction_set, abstraction_counts


def build_corpus_representations(corpus: List[List[str]]):
    """
    corpus: list of pieces (each piece = list of abstraction strings)
    
    Returns:
        sets: list of sets
        counts: list of Counters
    """
    sets = []
    counts = []
    
    for piece in corpus:
        s, c = extract_piece_abstractions(piece)
        sets.append(s)
        counts.append(c)
    
    return sets, counts



def jaccard_similarity(set_a: List[str], set_b: List[str])->float:
    if not set_a and not set_b:
        return 1.0  # define empty-empty as identical
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    return intersection / union


def top_k_abstractions(lib_corpus: List[List[str]], k: int) -> Tuple[List,List]:
    ...

def save_json(fpath:str, res:List|Dict):
    with open(fpath, "w") as f:
        json.dump(res, f, indent=2)
    print(f'Results saved to {fpath} !')
    
    
def load_json(fpath: str):
    assert os.path.isfile(fpath) # check file exist
    with open(fpath) as json_file:
        json_data = json.load(json_file)
    
    return json_data


if __name__=="__main__":
    user = os.path.expanduser("~")
    repo = f"{user}/codes/compression-similarity/"
   
    fjson = f"{repo}stitchCompression/stitch_res/corpus_stats.json"
    res = load_json(fjson)
    print(f'{res=}')