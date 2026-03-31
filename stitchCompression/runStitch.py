from dataclasses import dataclass
import json
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from stitch_core import Abstraction, compress, rewrite

@dataclass
class Corpus:
    titles: List[str]
    progressions: List[str]
    programs: List[str]

    @classmethod
    def from_json(cls, fpath: str) -> Corpus:
        """Load corpus metadata and programs from a JSON file."""
        with open(fpath) as f:
            corpus_data = json.load(f)
        return cls(
            titles=[x["pieceTitle"] for x in corpus_data],
            progressions=[x["progression"] for x in corpus_data],
            programs=[x["program"] for x in corpus_data],
        )

##  compress and rewrite programs 

def learned_abstractions_from_programs(programs:list[str],
                                        stitch_config: Dict
                                        )-> List[Abstraction]:
    """
    get abstractions for a set of programs.
    """
    ab = compress(programs=programs, **stitch_config).abstractions
    return ab

def rewrite_summary_with_abstractions(programs: List[str], abstractions: List[Abstraction]) -> Dict[str, Any]:
    """
    rewrite the a given set of programs with existing abstractions (library) and,
    return a dictionary containing summary stats/results of the rewrite operation.
    
    Returns:
        rewritten_programs : List[str]
        used_abstractions : List[Abstraction]
        rewritten_size : int
        used_library_size : int
        compressed_encoding_size: int (rewritten_size + used_library_size)
    """
    rewritten_res = rewrite(programs=programs, abstractions=abstractions, no_mismatch_check=True)
    rewritten_programs = rewritten_res.rewritten
    used_abs = rewritten_res.json["abstractions"]
    rewritten_size = sum(program_size(x) for x in rewritten_programs)
    used_lib_size = sum(program_size(ab["body"]) for ab in used_abs)
    return {
        "rewritten_programs": rewritten_programs,
        "used_abstractions": used_abs,
        "rewritten_size": rewritten_size,
        "used_library_size": used_lib_size,
        "compressed_encoding_size": rewritten_size + used_lib_size,
    }

## size and distance 

def program_size(program: str) -> int:
    """
    Program size = number of constructor tokens.
    """
    without_parentheses = program.replace("(", "").replace(")", "")
    return len(re.split(r"\s+", without_parentheses))

def encoding_size_with_abstractions(programs: List[str], abstractions: List[Abstraction]) -> int:
    """
    Compute the compressed encoding size of given programs using a precomputed abstraction library.
    """
    return rewrite_summary_with_abstractions(programs, abstractions)["compressed_encoding_size"]


def compressed_encoding_size(programs_test: List[str], programs_train: List[str], stitch_config: Dict) -> int:
    """
    Compute conditional (or cross) encoding size, C(test|train).

    The train corpus is compressed to produce abstractions, 
    then the test corpus is rewritten with that library.
    """
    abs_from_train = learned_abstractions_from_programs(programs_train, stitch_config)
    return encoding_size_with_abstractions(programs_test, abs_from_train)


def normalize_symmetric_compression_distance(p_x: List[str], p_y: List[str], stitch_config:Dict) -> float:
    """
    the normalized symmetric compreesion distance between two programs p_x and p_y,
    using the compressed_encoding_size
    
    """
    c_x = compressed_encoding_size(programs_test=p_x, programs_train=p_x, stitch_config=stitch_config)
    c_y = compressed_encoding_size(programs_test=p_y, programs_train=p_y, stitch_config=stitch_config)
    c_xy = compressed_encoding_size(programs_test=p_x, programs_train=p_y, stitch_config=stitch_config)
    c_yx = compressed_encoding_size(programs_test=p_y, programs_train=p_x, stitch_config=stitch_config)
    
    return max(c_xy, c_yx) / max(c_x, c_y)


## helper funcs
def save_json_file(fpath:str, res:List|Dict):
    with open(fpath, "w") as f:
        json.dump(res, f, indent=2)
    print(f'Results saved to {fpath} !')

## caching experiment data

def individual_program_abstractions(programs: List[str], stitch_config: Dict,
                                    save_json: Optional[str]=None) -> List[List[Abstraction]]:
    """
    Compress each program in the corpus independently and,
    return a list of list of Abstraction,
    save result to json file
    """
    
    res = [learned_abstractions_from_programs(programs=p, stitch_config=stitch_config) for p in programs]
    json_strings = [[str(abs_obj) for abs_obj in sublist] for sublist in res]

    if save_json:
        save_json_file(fpath=save_json, res=json_strings)
    return res

def pairwise_normalized_symmetric_compression_distance(corpus: Corpus, 
                                                       stitch_config: Dict, 
                                                       save_json: Optional[str] = None) -> pd.DataFrame:
    """
    Compute pairwise NSCD for all program pairs in the corpus.

    Returns a long-form dataframe with one row per (x, y) pair.
    """
    programs = corpus.programs
    titles = corpus.titles if corpus.titles else [f"program_{i}" for i in range(len(programs))]
    n = len(programs)

    if n == 0:
        empty_df = pd.DataFrame(columns=["i", "j", "x", "y", "nscd", "c_xy", "c_yx", "c_x", "c_y"])
        if save_json:
            with open(save_json, "w") as f:
                json.dump([], f, indent=2)
        return empty_df

    # compress each single program once
    abstractions_by_program = [
        learned_abstractions_from_programs(programs=[program], stitch_config=stitch_config)
        for program in programs
    ]

    # compute all directional encoding sizes C(i|j) and cache them as dictionary with (i,j) as key.
    single_programs = [[p] for p in programs]
    c_by_pair: Dict[tuple[int, int], float] = {
        (i, j): float(
            encoding_size_with_abstractions(
                programs=single_programs[i],
                abstractions=abstractions_by_program[j],
            )
        )
        for i in range(n)
        for j in range(n)
    }

    # self-compression terms C(i) are diagonal entries C(i|i).
    c_self = {i: c_by_pair[(i, i)] for i in range(n)}

    # build final pairwise records with NSCD.
    records: List[Dict[str, Any]] = []
    for i in range(n):
        for j in range(n):
            c_xy = c_by_pair[(i, j)]
            c_yx = c_by_pair[(j, i)]
            c_x = c_self[i]
            c_y = c_self[j]
            nscd = max(c_xy, c_yx) / max(c_x, c_y)

            records.append(
                {
                    "x": titles[i],
                    "y": titles[j],
                    "nscd": nscd,
                    "c_xy": c_xy,
                    "c_yx": c_yx,
                    "c_x": c_x,
                    "c_y": c_y,
                }
            )

    if save_json:
        with open(save_json, "w") as f:
            json.dump(records, f, indent=2)
        
        print(f'Individual library saved to {save_json} !')


    return pd.DataFrame(records)



## statistics
def corpus_stats(corpus:Corpus, stitch_config: Dict, save_json:Optional[str]) -> List[Dict[str,Any]]:
    """
    
    all stats before abstraction learning and rewrite
    program size dist
    library size dist
    """
    programs = corpus.programs
    titles = corpus.titles
    res: List[Dict[str, Any]] = []
    for i,x in enumerate(programs):
        p_size = program_size(program=x)     
        lib_size = len(learned_abstractions_from_programs(programs=[x], stitch_config=stitch_config))
        res.append(
            {
                "piece": titles[i],
                "program_size": p_size,
                "library_size":lib_size , 
            }
        )
    if save_json:
         save_json_file(fpath=save_json, res=res)
    return res       
        
## testing

def compress_rewrtie_behavior_test(testing_programs: List[str], stitch_config: Dict) -> None:
    """
    sanity check: helper-based size matches explicit rewrite + used library size.
    """
    
    cross_compression_function_size = compressed_encoding_size(
        programs_test=testing_programs,
        programs_train=testing_programs,
        stitch_config=stitch_config,
    )
    
    self_compression = compress(programs=testing_programs, **stitch_config)
    lib = self_compression.abstractions
    self_rewritten = rewrite(programs=testing_programs, abstractions=lib)
    
    rewritten_size = sum(program_size(x) for x in self_rewritten.rewritten)
    lib_size = sum(program_size(x["body"]) for x in self_rewritten.json["abstractions"])

    assert cross_compression_function_size == rewritten_size + lib_size
    
    
if __name__ == "__main__":
    user = os.path.expanduser("~")
    repo = f"{user}/codes/compression-similarity/"
    
    corpus = Corpus.from_json(fpath=f"{repo}corpus/converted_corpus_rep.json")
    all_programs = corpus.programs

    stitch_cfg = {
        "iterations": 100,
        "cost_app": 0,
        "cost_ivar": 1,
        "cost_lam": 0,
        "cost_prim_default": 1,
        "cost_var": 1,
        "allow_single_task": True,
        "no_mismatch_check": True,  # if this is False we will see an assertion error from Rust
    }
    
    # create result folder
    stitch_res_path = f'{repo}stitchCompression/stitch_res/'
    if not os.path.exists(stitch_res_path):
        os.makedirs(stitch_res_path)
    
    # individual_program_abstractions(programs=all_programs,
    #                                 stitch_config=stitch_cfg,
    #                                 save_json_path=f"{stitch_res_path}all_individual_library.json")
    
    # pairwise_normalized_symmetric_compression_distance(corpus=corpus, 
    #                                                    stitch_config=stitch_cfg, 
    #                                                    save_json=f"{stitch_res_path}similarity_stats.json")
    
    corpus_stats(corpus=corpus, stitch_config=stitch_cfg, save_json=f"{stitch_res_path}corpus_stats.json")