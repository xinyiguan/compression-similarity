from dataclasses import dataclass
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd
from stitch_core import Abstraction, compress, rewrite

from experiment.stitch_stats import get_corpus_stats

@dataclass
class Piece: 
    title: str 
    progression: List[str] 
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
            progression = x["progression"].split(" "),
            program = x["program"]
            ) for x in corpus_data]
                
        return cls(pieces = pieces[:50])
    
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
