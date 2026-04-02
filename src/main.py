from typing import Callable, Any

import preprocessing.data_formatting
import model.stitchCompression.runStitch
import experiment.pairwise_compression

def log(f):
    def _f (*args,**kwargs):
        print(f"[Start] {f.__module__}.{f.__name__}" )
        result = f(*args,**kwargs)
        print(f"[End] {f.__module__}.{f.__name__}" )
        return result
    return _f

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

if __name__ == "__main__":
    corpusPath = "data/corpus/preprocessed/converted_corpus_rep.json"
    # log(preprocessing.data_formatting.run)(
    #     path_treebank="data/corpus/original/treebank.json",
    #     path_output = "data/corpus/preprocessed/converted_corpus_rep.json"
    #     )
    # log(model.stitchCompression.runStitch.run)(
    #     path_preprocessed = "data/corpus/preprocessed/converted_corpus_rep.json",
    #     path_output= "data/result/stitch_res"
    # )
    experiment.pairwise_compression.run(
        corpus_path = corpusPath,
        output_path = "data/result/pairwise_similarity.json",
        stitch_config = stitch_cfg
    )
    

    
    