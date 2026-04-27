from typing import Dict

from model.utils import Corpus, Piece, save_json
from stitch_core import compress

def stitch_on_equinox_jody(corpus_path:str, stitch_config:Dict,
                           save_res: str):
    c = Corpus.from_json(corpus_path)

    equinox = c.get_piece(piece_name="(Valid)Equinox").program
    jody = c.get_piece(piece_name="(Valid)Jody Grind, The").program

    c_equinox = compress(programs=[equinox], **stitch_config)
    c_jody = compress(programs=[jody], **stitch_config)
    c_equinox_jody = compress(programs=[equinox] + [jody], **stitch_config)

    res = {
        "equinox": c_equinox.json,
        "jody": c_jody.json,
        "equinox_jody": c_equinox_jody.json
    }

    save_json(fpath=save_res, res=res)

    return res
