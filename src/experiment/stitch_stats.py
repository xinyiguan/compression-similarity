from typing import Any, Dict, List, Optional

import model.stitchCompression.runStitch
from src.model.utils import save_json


def get_corpus_stats(corpus:model.stitchCompression.runStitch.Corpus, 
                     stitch_config: Dict, save_json_path:Optional[str]) -> List[Dict[str,Any]]:
    """
    
    all stats before abstraction learning and rewrite
    Returns:
        - piece
        - program size
        - library size
        - learned library within piece 
    """
    programs = corpus.programs
    titles = corpus.titles
    res: List[Dict[str, Any]] = []
    for i,x in enumerate(programs):
        p_size = model.stitchCompression.runStitch.program_size(program=x)     
        learned_lib = model.stitchCompression.runStitch.learned_abstractions_from_programs(programs=[x], stitch_config=stitch_config)
        lib_strings = [str(abs_obj) for abs_obj in learned_lib]
        lib_size = model.stitchCompression.runStitch.library_size(lib=learned_lib)
        res.append(
            {
                "piece": titles[i],
                "program_size": p_size,
                "library_size":lib_size , 
                "library": lib_strings
            }
        )
    if save_json_path:
         save_json(fpath=save_json_path, res=res)
    return res       
        

def save_global_library(corpus:model.stitchCompression.runStitch.Corpus,
                        stitch_config: Dict,
                        save_json_path: str):
    g_lib = model.stitchCompression.runStitch.get_global_library(corpus=corpus,
                                                                 stitch_config=stitch_config)
    g_lib_str = [str(l) for l in g_lib]
    save_json(fpath=save_json_path, res=g_lib_str)