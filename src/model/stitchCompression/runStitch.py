from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from stitch_core import Abstraction, compress, rewrite

from model.stitchCompression.utils import save_json
from experiment.stitch_stats import get_corpus_stats

@dataclass
class Piece: 
    title: str 
    progression: str 
    program: str

@dataclass
class Corpus:
    pieces: List[Piece]    

    @classmethod
    def from_json(cls, fpath: str) -> Corpus:
        """Load corpus metadata and programs from a JSON file."""
        with open(fpath,'r') as f:
            corpus_data = json.load(f)
        pieces = [Piece(
            title = x["pieceTitle"],
            progression = x["progression"],
            program = x["program"]
            ) for x in corpus_data]
                
        return cls(pieces = pieces[:10])
    
    @property
    def programs(self) -> List[str]:
        return [x.program for x in self.pieces]
    
    @property
    def titles(self) -> List[str]:
        return [x.title for x in self.pieces]
    
    @property
    def progressions(self) -> List[str]:
        return [x.progressions for x in self.pieces]

##  compress and rewrite programs 

def learned_abstractions_from_programs(programs:list[str],
                                        stitch_config: Dict
                                        )-> List[Abstraction]:
    """
    get abstractions for a set of programs.
    """
    ab = compress(programs=programs, **stitch_config).abstractions
    return ab

def rewrite_with_abstractions(programs: List[str], abstractions: List[Abstraction]) -> Dict[str, Any]:
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

def library_size(lib: List[Abstraction]) -> int:
    """
    count the constructors in the rhs (the body)
    ["fn_0(#0,#1) := (#1 (#0 T T))",
     "fn_1(#0,#1) := (fn_0 #1 #0 T)"]
    """
    return sum([program_size(program=ab.body) for ab in lib])

def encoding_size_with_abstractions(programs: List[str], abstractions: List[Abstraction]) -> int:
    """
    Compute the compressed encoding size of given programs using a precomputed abstraction library.
    """
    return rewrite_with_abstractions(programs, abstractions)["compressed_encoding_size"]

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

## caching experiment data
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

    print(f'learning the individual libraries ...')
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
    print(f'builidng records for piece {i} ...')
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

def get_global_library(corpus:Corpus, stitch_config: Dict) -> List[Abstraction]:
    """
    get the global library
    """
    return compress(programs=corpus.programs, **stitch_config).abstractions
    

## experiments

def rewrite_individual_using_global_lib(corpus:Corpus, 
                                        stitch_config: Dict, 
                                        save_json_path:Optional[str]) -> List[Dict[str,Any]]:
    all_programs = corpus.programs
    titles = corpus.titles
    g_lib = get_global_library(corpus=corpus, stitch_config=stitch_config, save_json_path=None)
    res: List[Dict[str, Any]] = []
    for i,x in enumerate(all_programs):
        rewrite_res = rewrite_with_abstractions(programs=[x], abstractions=g_lib)
        rewrite_res["piece"] = titles[i]
        res.append(rewrite_res)
    if save_json_path:
        save_json_file(fpath=save_json_path, res=res)
    return res



def run(path_preprocessed:str,path_output:str):
    corpus = Corpus.from_json(fpath=path_preprocessed)

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
    

    print(f'Computing corpus statistics ...')
    get_corpus_stats(corpus=corpus, stitch_config=stitch_cfg, save_json_path=f"{path_output}/corpus_stats.json")
    
    print(f'Computing the global library ...')
    get_global_library(corpus=corpus,stitch_config=stitch_cfg)
    
    # print(f'Running pairwise similarity measure ...)')
    # pairwise_normalized_symmetric_compression_distance(corpus=corpus, 
    #                                                    stitch_config=stitch_cfg, 
    #                                                    save_json_path=f"{path_output}/pairwise_similarity_stats.json")
    