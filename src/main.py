from typing import List, Literal, Tuple

import experiment.pairwise_compression
import experiment.pairwise_compression_corpus
import experiment.edit_distance
from model.utils import load_json,Corpus
from experiment.report import corr_rank_report
import docGen.main
from experiment.clustering import cluster_edit_distances, cluster_test
from experiment.case_study import stitch_on_equinox_jody
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


def main():
    corpusPath = "data/corpus/preprocessed/converted_corpus_rep.json"

    recompute_edit_distance = False 
    if recompute_edit_distance:
        experiment.edit_distance.run(
            corpus_path=corpusPath,
            output_path=f"data/result/edit_distances(normalized).json", save_result=True
        )

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
        cluster_edit_distances("data/result")


if __name__ == "__main__":
    # main()
    corpusPath = "data/corpus/preprocessed/converted_corpus_rep.json"

    stitch_on_equinox_jody(corpus_path=corpusPath,stitch_config=stitch_cfg,
                           save_res=f'data/result/stitch(equinox_jody).json'
                           )
