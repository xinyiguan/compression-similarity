from collections import ChainMap, defaultdict
from dataclasses import dataclass
from functools import partial
from itertools import batched, combinations
import json
from typing import IO, Any, Callable, Dict, List, NamedTuple, Optional, Tuple

from more_itertools import flatten
from tqdm import tqdm
from model.stitchCompression.runStitch import (
    Corpus,
    get_global_library,
    rewrite_with_abstractions,
    stitch_compress,
)
from model.utils import Piece, load_json, save_json
import model.PPMCompression.ppm_compression as ppm
from multiprocessing import Pool
import os


class Model[a](NamedTuple):
    name: str  # model name (i.e. "ppm", "stitch")
    # c_xy: Callable[[a, a], float]  # the model's C function as in C(xy)
    c: Callable[[a], float]  # the model's C function as in C(x) or C(y)


type PieceName = str
type ModelName = str


def pairwise_NCD(c_result: CResult[PieceName], model_names: List[str]) -> List[Dict]:
    json = []
    for x, y in c_result.pairwise_C_ctx.keys():

        entry = {"nameX": x.replace('(Valid)',''), "nameY": y.replace('(Valid)','')}
        for model_name in model_names:
            c_xy_ctx = c_result.get_pairwise_C_ctx(x, y, model_name)
            c_x_ctx = c_result.get_individual_C_ctx(x, model_name)
            c_y_ctx = c_result.get_individual_C_ctx(y, model_name)
            c_x = c_result.get_individual_C(x, model_name)
            c_y = c_result.get_individual_C(y, model_name)
            d_ctx = (c_xy_ctx - min(c_x_ctx, c_y_ctx)) / max(c_x, c_y)
            entry.update(
                {
                    f"Cxy_ctx_{model_name}": c_xy_ctx,
                    f"Cx_ctx_{model_name}": c_x_ctx,
                    f"Cy_ctx_{model_name}": c_y_ctx,
                    f"Cx_{model_name}": c_x,
                    f"Cy_{model_name}": c_y,
                    f"D_{model_name}": d_ctx,
                }
            )
        json.append(entry)
    return json


@dataclass
class CResult[p]:
    """
    CResult represents memoized result of the C values of all models
    It uses nested dictionary (as opposed to flat list records) for efficient access.

    type parameter `p` represent piece id
        - e.g. str
        - `p` needs to be hashable to be used dictionary key
    """

    individual_C: Dict[p, Dict[ModelName, float]]
    individual_C_ctx: Dict[p, float]
    pairwise_C_ctx: Dict[Tuple[p, p], Dict[ModelName, float]]

    def get_individual_C(self, p: p, model_name: ModelName) -> float:
        return self.individual_C.get(p).get(model_name)

    def get_individual_C_ctx(self, p: p, model_name: ModelName) -> float:
        return self.individual_C_ctx.get(p).get(model_name)

    def get_pairwise_C_ctx(self, x: p, y: p, model_name: ModelName) -> float:
        return self.pairwise_C_ctx.get((x, y)).get(model_name)

    def to_json(self) -> Dict[str, Any]:
        return {
            "individual_C": self.individual_C,
            "individual_C_ctx": self.individual_C_ctx,
            "pairwise_C_ctx": {str(k): v for k, v in self.pairwise_C_ctx.items()},
        }


def action_single_c(input, models, ctx):
    i, piece = input
    # print(f"single piece c: {i}/{total_size}")
    updates = []
    for model in models:
        updates.append(((("individual_C", piece.title, model.name), model.c([piece]))))
        updates.append(
            (("individual_C_ctx", piece.title, model.name), model.c([piece] + ctx))
        )

    return updates


def action_pair_c(input, models, ctx):
    (x, y) = input
    # print(f"pairwise c (with ctx): {i}/{total_size}")
    updates = []
    for model in models:
        updates.append(
            (
                ("pairwise_C_ctx", (x.title, y.title), model.name),
                model.c([x, y] + ctx),
            )
        )

    return updates


def precompute_c_vals[p](
    all_pieces: List[p],
    ctx: List[p],
    models: List[Model[p]],
    output_folder: str
) -> CResult[p]:

    process_single_C = False
    process_pairs_C = False

    if process_single_C:
        # compute single piece's C value (with and without context)
        print("computing c for single pieces with/without context")

        all_pieces_size = len(all_pieces)
        with Pool(os.cpu_count() - 1) as pool:
            updatess = list(
                tqdm(
                    pool.imap_unordered(
                        partial(
                            action_single_c,
                            models=models,
                            ctx=ctx,
                        ),
                        enumerate(all_pieces),
                    ),
                    total=all_pieces_size,
                )
            )
        updates = list(flatten(updatess))
        save_json(f"{output_folder}/temp/Cx_updates.json", updates)

    # merge list of updates into dictionary for efficient access
    updates = load_json(f"{output_folder}/temp/Cx_updates.json")
    standaloneC = defaultdict(dict)
    c_with_context = defaultdict(dict)
    for update in updates:
        # print(update)
        (mode, title, model_name), val = update
        if mode == "individual_C":
            standaloneC[title][model_name] = val
        if mode == "individual_C_ctx":
            c_with_context[title][model_name] = val

    save_json(f"{output_folder}/temp/standaloneC.json", standaloneC)
    save_json(f"{output_folder}/temp/c_with_context.json", c_with_context)

    if process_pairs_C:
        # compute pairs' C value (with and without context)
        pairs = [(p, q) for p, q in combinations(all_pieces, 2) if p.title != q.title]

        print("computing c for pairs with/without context")
        with Pool(os.cpu_count() - 1) as pool:
            pair_updatess = list(
                tqdm(
                    pool.imap_unordered(
                        partial(action_pair_c, models=models, ctx=ctx), pairs
                    ),
                    total=len(pairs),
                )
            )
        pair_updates = list(flatten(pair_updatess))
        save_json(f"{output_folder}/temp/Cpair_updates.json", pair_updates)

    # merge list of updates into dictionary for efficient access
    pair_updates = load_json(f"{output_folder}/temp/Cpair_updates.json")
    c_pair_with_context = defaultdict(dict)
    for update in pair_updates:
        (mode, pair, model_name), val = update
        if mode == "pairwise_C_ctx":
            c_pair_with_context[tuple(pair)][model_name] = val

    result = CResult(
        individual_C=standaloneC,
        individual_C_ctx=c_with_context,
        pairwise_C_ctx=c_pair_with_context,
    )
    # print(result)
    save_json(f"{output_folder}/c_result.json", result.to_json())
    return result


def stitchC(xs, stitch_config):
    return stitch_compress(programs=Corpus(xs).programs, stitch_config=stitch_config)[
        "compressed_encoding_size"
    ]


def ppmC(xs, alphabet_size, aggregation):
    return ppm.compress(
        target_seqs=Corpus(xs).progressions,
        training_seqs=Corpus(xs).progressions,
        alphabet_size=alphabet_size,
        aggregation=aggregation,
    )


def run(
    corpus_path: str,
    output_folder: str,
    with_ctx: bool,
    stitch_config: Dict,
):
    """
    run the pairwise similarity measure (NCD) for each pair in the corpus, and
    return 3 json files: stitch-relevant results, ppm-relevant results, pairwise C & D vals
    """

    corpus = Corpus.from_json(fpath=corpus_path)

    # hyper parameters
    alphabet_size = 311
    aggregation = sum
    
    all_pieces = corpus.pieces

    # configure models
    models = [
        Model(
            "stitch",
            partial(stitchC, stitch_config=stitch_config),
        ),
        Model(
            "ppm",
            partial(
                ppmC,
                alphabet_size=alphabet_size,
                aggregation=aggregation,
            ),
        ),
    ]

    # do heavy works
    if with_ctx:
        ctx = corpus.pieces
    else:
        ctx = []
    c_result = precompute_c_vals(
        all_pieces=all_pieces, 
        ctx=ctx, 
        models=models,
        output_folder = output_folder
        )

    # compute NCD using cached c values
    print(f"Computing NCD ...")
    result_json = pairwise_NCD(c_result=c_result, model_names=[m.name for m in models])

    save_json(f"{output_folder}/pairwise_similarity.json", result_json)
