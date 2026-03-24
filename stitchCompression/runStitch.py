from dataclasses import dataclass
import json, os,re

from typing import List, Dict, Self

from stitch_core import compress, rewrite


def corpus_total_size(corpus:json) -> int:
    raise NotImplementedError

def compress_size_by(x: List[str], y: List[str]) -> float:
    pattern_y = compress(y).abstractions
    return (rewrite(x,pattern_y))


@dataclass
class Corpus:
    titles: List[str]
    progressions: List[str]
    programs: List[str]

    @classmethod
    def from_json(cls, fpath:str) -> Corpus:
        with open(fpath) as f:
            c = json.load(f)
        ts = [x['pieceTitle'] for x in c]
        progressions = [x['progression'] for x in c]
        programs = [x['program'] for x in c]
        instance = cls(titles=ts, progressions=progressions, programs=programs)
        return instance

def program_size(program:str):
    removed_parathesis = program.replace("(","").replace(")","")
    constructors = re.split(r'\s+',removed_parathesis)
    result = len(constructors)
    return result



def compressed_encoding_size(corpus_test: List[str], corpus_train: List[str], stitch_config:Dict) -> float:
    """
    compress a corpus of program (corpus_train) to extract a reusble library, and
    rewrite another corpus (corpus_test) using the library
    """
    
    compressed_train = compress(programs=corpus_train, **stitch_config)
    lib_train = compressed_train.abstractions
    rewritten_test = rewrite(programs=corpus_test, abstractions=lib_train)
    
    size_rewritten = sum([program_size(x) for x in rewritten_test.rewritten])
    size_used_abs = sum([program_size(ab['body']) for ab in rewritten_test.json['abstractions']])
    result = size_rewritten + size_used_abs
    
    print(result)
    return result


if __name__=="__main__":
    user = os.path.expanduser("~")
    repo = f'{user}/codes/compression-similarity/'

    programs = Corpus.from_json(fpath=f"{repo}CorpusRep.json").programs
    
    stitch_config = {
    "iterations":100,    
    "cost_app": 0, 
    "cost_ivar" : 1, 
    "cost_lam" : 0,
    "cost_prim_default" :1,
    "cost_var" : 1,
    "allow_single_task":True,
    "no_mismatch_check":True # if this is False we will see an assertion error from Rust
}
    corpus_train = programs[:100]
    corpus_test = programs[101:102]
    
    compressed_encoding_size(corpus_test=corpus_test, corpus_train=corpus_train, stitch_config=stitch_config)
    
    