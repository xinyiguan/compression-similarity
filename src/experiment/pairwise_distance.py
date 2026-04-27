from dataclasses import dataclass
from itertools import combinations
import re
from typing import List, Literal, Optional, Sequence, Tuple
import Levenshtein
from zss import simple_distance, Node
from apted.helpers import Tree as APTEDNode
from apted import APTED,Config

from model.utils import Corpus, Piece, save_json

def string_edit_distance(s1: str, s2: str) ->float:
    return Levenshtein.distance(s1, s2)

def tree_edit_distance(p1: str, p2: str, method: Literal["zss", "apted"]) -> float:
    """
    Compute tree edit distance using Zhang-Shasha algorithm via zss.
    takes 2 programs (str), first turn them into tree object, then compute
    """
    # programs -> trees
    t1 = Tree.from_program(p1)
    t2 = Tree.from_program(p2)

    if method == "zss":
        # Tree -> zss format
        z1 = t1.to_zss()
        z2 = t2.to_zss()
        return simple_distance(z1, z2)
    
    elif method == "apted":
        # Tree -> apted trees
        a1 = t1.to_apted()
        a2 = t2.to_apted()
        apted = APTED(a1, a2, Config())  # default config: unit costs
        return apted.compute_edit_distance()

def run(corpus_path: str,
    output_path: Optional[str],
    save_result: bool
    ):
    """
    run the 3 types of pairwise distance for each pair in the corpus. 
    return 1 json file containing the distance values for each pair.
    """
    corpus = Corpus.from_json(fpath=corpus_path)

    res = []

    for i, pieceX in enumerate(corpus.pieces):
        print(f'Computing distances for {i}-th piece')
        progX = pieceX.program
        
        for pieceY in corpus.pieces:
            progY = pieceY.program
            res.append(
                {
                    "nameX": pieceX.title,
                    "nameY": pieceY.title,
                    "SED": string_edit_distance(pieceX.progression, pieceY.progression),
                    "TED(zss)": tree_edit_distance(progX, progY, "zss"),
                }
            )

    if save_result:
        if output_path is None:
            raise ValueError("output_path must not be None")
        save_json(fpath=output_path, res=res)
    return res


@dataclass(frozen=True)
class Label:
    """
    Structured representation of a node label.

    Example:
        rightHeaded-5-7-m7 →
            direction = "rightHeaded"
            degree = 5
            quality = ("7", "m7")
    """

    direction: str
    degree: Optional[int]
    quality: Tuple[str, ...]

    @classmethod
    def from_token(cls, lable_str: str) -> "Label":
        """
        create the structured `Label` object from string
        """
        if lable_str == "T":
            return cls(direction="T", degree=None, quality=())

        parts = lable_str.split("-")
        direction = parts[0]

        try:
            degree = int(parts[1])
            quality = tuple(parts[2:])
        except (IndexError, ValueError):
            degree = None
            quality = tuple(parts[1:])

        return cls(direction=direction, degree=degree, quality=quality)


@dataclass(frozen=True)
class Tree:
    """
    Immutable tree node.

    - label: structured Label
    - children: ordered list of subtrees
    """

    label: Label
    children: Tuple["Tree", ...]

    @classmethod
    def from_program(cls, program:str) ->"Tree":
        """
        Parse a program string into a fully structured Tree.
        """
        tokens = cls._tokenize(program=program)
        tree, remaining = cls._recursive_parse(tokens)

        if remaining:
            raise ValueError(f"Unparsed tokens: {remaining}")

        return tree

    @staticmethod
    def _tokenize(program:str) -> List[str]:
        """
        Split program string into tokens:
        - '('
        - ')'
        - labels
        """
        return re.findall(r'\(|\)|[^\s()]+', program)

    @classmethod
    def _recursive_parse(cls, tokens: Sequence[str]) -> Tuple[Tree, Sequence[str]]:
        """
        recursively parse a sequence of tokens to (parsed_tree, remaining_tokens).

        subtree: (rightHeaded-0-m7-m7 ...
        terminal/leaf: T
        """

        t = tokens[0]

        # case: subtree
        if t == "(":
            label = Label.from_token(tokens[1])

            def parse_children(
                rest: Sequence[str],
            ) -> Tuple[Tuple["Tree", ...], Sequence[str]]:
                return (
                    ((), rest)
                    if rest[0] == ")"
                    else (
                        lambda child, rest1: (
                            lambda others, rest2: ((child,) + others, rest2)
                        )(*parse_children(rest1))
                    )(*cls._recursive_parse(rest))
                )

            children, rest = parse_children(tokens[2:])
            return cls(label, children), rest[1:]  # skip ')'

        # case: leaf
        else:
            return cls(Label.from_token(t), ()), tokens[1:]

    @staticmethod
    def _label_to_string(label: Label) -> str:
        """
        convert Label into a clean comparable string.
        """
        parts = [label.direction]

        if label.degree is not None:
            parts.append(str(label.degree))

        parts.extend(label.quality)
        res = "-".join(parts)
        return res

    def to_zss(self) -> Node:
        """
        Convert our Tree into zss.Node
        """
        label_str = self._label_to_string(self.label)
        node = Node(label_str)

        for child in self.children:
            node.addkid(child.to_zss())

        return node

    def to_apted(self) -> APTEDNode:
        """
        Convert our Tree into apted.helpers.Tree
        """
        label_str = self._label_to_string(self.label)
        res = APTEDNode(label_str, [child.to_apted() for child in self.children])
        return res

def compute_edit_distance(corpus:Corpus[Piece]):
    result = DefaultDict(dict)
    pairs = [(p, q) for p, q in combinations(corpus.pieces, 2) if p.title != q.title]
    for x,y in pairs:
        for dist_type in ['str_edit','tree_edit']: 
            result[x.title][y.title][dist_type] = ...
    
        