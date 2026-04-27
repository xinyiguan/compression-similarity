from typing import List, Literal, Tuple

import experiment.pairwise_compression
import experiment.pairwise_compression_corpus
import experiment.edit_distance
from model.utils import load_json,Corpus
from experiment.report import corr_rank_report
import docGen.main
from experiment.clustering import cluster_test
def log(f):
    def _f(*args, **kwargs):
        print(f"[Start] {f.__module__}.{f.__name__}")
        result = f(*args, **kwargs)
        print(f"[End] {f.__module__}.{f.__name__}")
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


    # corr_rank_report(
    #     res_file=res,
    #     models=("stitch", "ppm"),
    #     save_report_fpath=f"data/result/report(stitch_ppm).json",
    #     show_results=True
    # )

    # assert False

    # data = load_json(res_path)
    # d_stitch = [e["D_stitch"] for e in data]
    # d_ppm= [e["D_ppm"] for e in data]

    # r = pearsonr(x=d_stitch, y=d_ppm)
    # print(f'{r=}')
    # assert False

    # log(preprocessing.data_formatting.run)(
    #     path_treebank="data/corpus/original/treebank.json",
    #     path_output = "data/corpus/preprocessed/converted_corpus_rep.json"
    #     )
    # log(model.stitchCompression.runStitch.run)(
    #     path_preprocessed = "data/corpus/preprocessed/converted_corpus_rep.json",
    #     path_output= "data/result/stitch_res"
    # )

    # pair-wise cross compression

    experiment.edit_distance.run(
        corpus_path=corpusPath,
        output_path=f"data/result/edit_distances(normalized).json", save_result=True
    )
    assert False
    for with_ctx in [False, True]:

        out_folder = f"data/result/(ctx={with_ctx})"
        experiment.pairwise_compression_corpus.run(
            corpus_path=corpusPath,
            output_folder=out_folder,
            stitch_config=stitch_cfg,
            with_ctx=with_ctx,
        )

        similarity_file = f"{out_folder}/pairwise_similarity.json"
        regression_topk_file = f"{out_folder}/regression_and_topk.json"

        corr_rank_report(
            res_file=similarity_file,
            model_x="ppm",
            model_y="stitch",
            save_report_fpath=regression_topk_file,
            show_results=False,
        )

        docGen.main.main(
            table_json_path = regression_topk_file,
            heatmap_vega_path = f"docs/vega/(ctx={with_ctx})/pairwise-similarity.vega",
            output_folder = f"docs/latex/figs/(ctx={with_ctx})"
        )
        
        cluster_test("data/result")
        
        
        
