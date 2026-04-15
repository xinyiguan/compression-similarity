from dataclasses import dataclass
import json
import os
from typing import Dict, List


@dataclass
class Piece:
    title: str
    progression: List[str]
    program: str

    def to_json(self) -> Dict:
        return {
            "title": self.title,
            "progression": self.progression,
            "program": self.program,
        }


@dataclass
class Corpus:
    pieces: List[Piece]

    @classmethod
    def from_json(cls, fpath: str) -> Corpus:
        """Load corpus metadata and programs from a JSON file."""
        with open(fpath, "r") as f:
            corpus_data = json.load(f)
        pieces = [
            Piece(
                title=x["pieceTitle"],
                progression=x["progression"].split(" "),
                program=x["program"],
            )
            for x in corpus_data
        ]

        return cls(pieces=pieces)

    @property
    def programs(self) -> List[str]:
        return [x.program for x in self.pieces]

    @property
    def titles(self) -> List[str]:
        return [x.title for x in self.pieces]

    @property
    def progressions(self) -> List[str]:
        return [x.progressions for x in self.pieces]


def save_json(fpath:str, res:List|Dict):
    dirpath = os.path.dirname(fpath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    with open(fpath, "w") as f:
        json.dump(res, f, indent=2)
    print(f'Results saved to {fpath} !')


def load_json(fpath: str):
    assert os.path.isfile(fpath) # check file exist
    with open(fpath) as json_file:
        json_data = json.load(json_file)
    
    return json_data


if __name__=="__main__":
    user = os.path.expanduser("~")
    repo = f"{user}/codes/compression-similarity/"
   
    fjson = f"{repo}stitchCompression/stitch_res/corpus_stats.json"
    res = load_json(fjson)
    print(f'{res=}')
