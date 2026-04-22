from typing import List, Literal, Tuple

import experiment.pairwise_compression
import experiment.pairwise_distance
from experiment.topK import get_topK_similar_pair, pprint_topK_table
from model.utils import load_json
from experiment.report import corr_rank_report

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


def topK_pairs_tables(res_file: str, k:int, topK_similiar:bool=True, save_latex:bool=True):

    k_ppm = get_topK_similar_pair(res_path=res_file, model="ppm", k=20)
    k_stitch = get_topK_similar_pair(res_path=res_file, model="stitch", k=20)

    if save_latex:
        pp_ppm = pprint_topK_table(topK_res=k_ppm, save_fpath="data/result/top20_ppm.txt")
        pp_stitch = pprint_topK_table(topK_res=k_stitch, save_fpath="data/result/top20_stitch.txt")

    return pp_ppm, pp_stitch


if __name__ == "__main__":
    corpusPath = "data/corpus/preprocessed/converted_corpus_rep.json"
    res="data/result/unique_pairwise_similarity.json"

    corr_rank_report(
        res_file=res,
        models=("stitch", "ppm"),
        save_report_fpath=f"data/result/report(stitch_ppm).json",
        show_results=True
    )

    assert False
    r, p = corr(res_file=res_path, models=["ppm", "stitch"], method="spearman")

    # data = load_json(res_path)
    # d_stitch = [e["D_stitch"] for e in data]
    # d_ppm= [e["D_ppm"] for e in data]

    # r = pearsonr(x=d_stitch, y=d_ppm)
    # print(f'{r=}')
    # assert False

    assert False

    # log(preprocessing.data_formatting.run)(
    #     path_treebank="data/corpus/original/treebank.json",
    #     path_output = "data/corpus/preprocessed/converted_corpus_rep.json"
    #     )
    # log(model.stitchCompression.runStitch.run)(
    #     path_preprocessed = "data/corpus/preprocessed/converted_corpus_rep.json",
    #     path_output= "data/result/stitch_res"
    # )

    # pair-wise cross compression
    experiment.pairwise_compression.run(
        corpus_path=corpusPath,
        output_path="data/result/pairwise_similarity.json",
        stitch_config=stitch_cfg,
        save_results=True,
        stitch_res_path="data/result/pairwise_stitch_compression.json",
        ppm_res_path="data/result/pairwise_ppm_compression.json"
    )
    # experiment.pairwise_compression.get_self_similarity_result(d_res_fpath="data/result/pairwise_similarity.json",
    #                                                            stitch_res_fpath="data/result/pairwise_stitch_compression.json",
    #                                                            save_d_fpath="data/result/self_similarity.json",
    #                                                            save_stitch_fpath="data/result/self_stitch_compression.json")

    # # top k similar pairs:
    # ppm_topK_similar = get_topK_similar_pair(res_path="data/result/pairwise_similarity.json", k=15, model="ppm",
    #                                          save_fpath="data/result/topK_ppm_similar_pairs.json")
    # stitch_topK_similar = get_topK_similar_pair(res_path="data/result/pairwise_similarity.json", k=15, model="stitch",
    #                                             save_fpath="data/result/topK_stitch_similar_pairs.json")

    # test

    # model.stitchCompression.runStitch.test(corpusPath=corpusPath, stitch_config=stitch_cfg)
