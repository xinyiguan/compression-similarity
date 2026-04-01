from typing import Callable, Any

import preprocessing.data_formatting
import model.stitchCompression.runStitch

def log(f):
    def _f (*args,**kwargs):
        print(f"[Start] {f.__module__}.{f.__name__}" )
        result = f(*args,**kwargs)
        print(f"[End] {f.__module__}.{f.__name__}" )
        return result
    return _f

if __name__ == "__main__":
    log(preprocessing.data_formatting.run)(
        path_treebank="data/corpus/original/treebank.json",
        path_output = "data/corpus/preprocessed/converted_corpus_rep.json"
        )
    log(model.stitchCompression.runStitch.run)(
        path_preprocessed = "data/corpus/preprocessed/converted_corpus_rep.json",
        path_output= "data/result/stitch_res"
    )
    

    
    