import json
from stitch_core import compress

with open('CorpusRep.json') as f:
    corpus = json.load(f)
programs =  [x['program'] for x in corpus]

stitch_config = {
    "cost_app": 0, 
    "cost_ivar" : 1, 
    "cost_lam" : 0,
    "cost_prim_default" :1,
    "cost_var" : 1,
    "allow_single_task":True,
    "no_mismatch_check":True # if this is False we will see an assertion error from Rust
}


def minimal_problematic_case():
    return compress([programs[49]],iterations=100,**stitch_config)

def compress_whole_corpus():
    return compress(programs,iterations=100,**stitch_config)


if __name__=="__main__":
    print(f"simple case result: {minimal_problematic_case().abstractions}")
    print(f"whole corpus case: {compress_whole_corpus().abstractions}")
