from itertools import combinations
from typing import Any, Callable, Dict, List, NamedTuple, Optional
from model.stitchCompression.runStitch import Corpus, rewrite_with_abstractions
from model.utils import load_json, save_json
import model.PPMCompression.ppm_compression as ppm


class Model[a](NamedTuple):
    name: str  # model name (i.e. "ppm", "stitch")
    c_xy: Callable[[a, a], float]  # the model's C function as in C(xy)
    c: Callable[[a], float]  # the model's C function as in C(x) or C(y)


def pairwise_NCD[a](
    corpus: List[a],
    models: List[Model[a]],
    title_of: Callable[[a], str],
) -> List[Dict]:

    json = [
        build_record(x, y, models=models, title_of=title_of)
        # for x in corpus
        # for y in corpus
        for x, y in combinations(corpus, 2)
        if title_of(x) != title_of(y)
    ]
    return json


def build_record[a](
    x: a, y: a, models: List[Model[a]], title_of: Callable[[a], str]
) -> Dict:

    result = {"nameX": title_of(x), "nameY": title_of(y)}
    for model_name, c_xy, c in models:
        c_xy_val = c_xy(x, y)
        c_x_val = c(x)
        c_y_val = c(y)
        d_val = (c_xy_val - min(c_x_val, c_y_val)) / max(c_x_val, c_y_val)
        result.update(
            {
                f"Cxy_{model_name}": c_xy_val,
                f"Cx_{model_name}": c_x_val,
                f"Cy_{model_name}": c_y_val,
                f"D_{model_name}": d_val,
            }
        )
    return result


def run(
    corpus_path: str,
    output_path: str,
    stitch_config: Dict,
    save_results: bool,
    stitch_res_path: Optional[str],
    ppm_res_path: Optional[str]
):
    """
    run the pairwise similarity measure (NCD) for each pair in the corpus, and
    return 3 json files: stitch-relevant results, ppm-relevant results, pairwise C & D vals
    """

    corpus = Corpus.from_json(fpath=corpus_path)
    stitchResults = []
    ppmResults = []
    alphabet_size = 311
    aggregation = sum
    print(f'Computing C_xy ...')

    for i, x in enumerate(corpus.pieces):
        print(f"Running {i}-th piece...")
        for y in corpus.pieces:

            # STITCH
            stitchResults.append(
                {
                    "pieceX": x.title,
                    "pieceY": y.title,
                    "stitch_xy_res": rewrite_with_abstractions(
                        programs_test=[x.program] + [y.program],
                        programs_train=[x.program] + [y.program],
                        stitch_config=stitch_config,
                    ),  # C_stitch(xy) := rewrite (xy, pat(xy))
                }
            )

            # PPM
            ppmResults.append(
                {
                    "pieceX": x.title,
                    "pieceY": y.title,
                    # C_ppm(xy) :=  compress([x, y], [x, y])
                    "ppm_Cxy": ppm.compress(
                        target_seqs=[x.progression, y.progression],
                        training_seqs=[x.progression, y.progression],
                        alphabet_size=alphabet_size,
                        aggregation=aggregation
                    ),
                }
            )

    stitch_Cxy = lambda x, y: [
        entry["stitch_xy_res"]["compressed_encoding_size"]
        for entry in stitchResults
        if entry.get("pieceX") == x.title and entry.get("pieceY") == y.title
    ][0]

    ppm_Cxy = lambda x, y: [
        entry["ppm_Cxy"]
        for entry in ppmResults
        if entry["pieceX"] == x.title and entry["pieceY"] == y.title
    ][0]

    print(f"Computing stitch C_x ...")
    stitch_C = lambda x: rewrite_with_abstractions(
        programs_test=[x.program],
        programs_train=[x.program],
        stitch_config=stitch_config,
    )["compressed_encoding_size"]

    print(f"Computing ppm C_x ...") 
    # C(x) -->  compress([x], [x])
    ppm_C = lambda x: ppm.compress(
        target_seqs=[x.progression],
        training_seqs=[x.progression],
        alphabet_size=alphabet_size,
        aggregation=aggregation
    )

    print(f'Computing NCD ...')
    json = pairwise_NCD(
        corpus=corpus.pieces,
        models=[
            Model("stitch", stitch_Cxy, stitch_C), 
            Model("ppm", ppm_Cxy, ppm_C)],
        title_of=lambda x: x.title,
    )

    if save_results:
        if ppm_res_path is None or stitch_res_path is None:
            raise ValueError("ppm_res_path and stitch_res_path must not be None")
        save_json(stitch_res_path, stitchResults)
        save_json(ppm_res_path, ppmResults)
        save_json(output_path, json)


def get_self_similarity_result(
    d_res_fpath: str, stitch_res_fpath: str, save_d_fpath: str, save_stitch_fpath: str
):
    d_res = load_json(fpath=d_res_fpath)
    stitch_res = load_json(fpath=stitch_res_fpath)

    self_d = [entry for entry in d_res if entry["nameX"] == entry["nameY"]]
    self_stitch = [entry for entry in stitch_res if entry["pieceX"] == entry["pieceY"]]

    save_json(fpath=save_d_fpath, res=self_d)
    save_json(fpath=save_stitch_fpath, res=self_stitch)


def unique_pairs_result(res_path: str, exclude_self: bool=False, 
                        save_json_path: Optional[str]=False):
    data: List[Dict[str, Any]] = load_json(res_path)
    seen = set()
    result = []

    for item in data:
        key = tuple(sorted([item["nameX"], item["nameY"]]))

        if key not in seen:
            seen.add(key)
            result.append(item)

    if exclude_self:
        result = [r for r in result if r["nameX"]!=r["nameY"]]

    if save_json_path:
        assert isinstance(save_json_path, str)
        save_json(fpath=save_json_path, res=result)

    return result
