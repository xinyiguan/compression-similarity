from typing import Callable, Dict, List, Tuple

# from model.PPMCompression.ppm_compression_distance import compress_seq
from model.similarity import compression_distance
from model.stitchCompression.runStitch import Corpus, compressed_encoding_size
from model.stitchCompression.utils import save_json

    
def pairwise_NCD[a](
    corpus: List[a], 
    models: List[Tuple[
        str, # model name (i.e. "ppm", "stitch")
        Callable[[a,a],float] # the model's C function as in C(x|y)
        ]],
    title_of: Callable[[a], str]
    ) -> List[Dict]:
    json = [build_record(x,y,models=models,title_of=title_of) for x in corpus for y in corpus]
    return json

def build_record[a](
    x:a,
    y:a,
    models: List[Tuple[
        str, # model name (i.e. "ppm", "stitch")
        Callable[[a,a],float] # the model's C function as in C(x|y)
        ]],
    title_of: Callable[[a], str]
    ) -> Dict:
    result = {
        "nameX": title_of(x),
        "nameY": title_of(y), 
        } 
    for (model_name,c_func) in models:
        c_val = c_func(x,y)
        d_val = compression_distance(c_func)(x,y)
        result.update({
            f"C_{model_name}":c_val,
            f"D_{model_name}":d_val
            })
    return result


def run(corpus_path:str,output_path:str, stitch_config):
    corpus = Corpus.from_json(fpath=corpus_path)
    json = pairwise_NCD(
        corpus=corpus.pieces,
        models=[
            ("stitch",
             lambda x,y: compressed_encoding_size([x.program],[y.program],stitch_config = stitch_config))
            # ("ppm",
            #  lambda x,y: compress_seq(seq=x,pretrain_seq=y,alphabet_size=300)
            #  )
        ],
        title_of= lambda x: x.title)
    print(json)
    save_json(output_path,json)