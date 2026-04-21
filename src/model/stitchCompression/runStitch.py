import re
from typing import Any, Dict, List
from stitch_core import Abstraction, compress, rewrite

from model.utils import Corpus


# size
def program_size(program: str) -> int:
    """
    Program size = number of constructor tokens.
    """
    without_parentheses = program.replace("(", "").replace(")", "")
    return len(re.split(r"\s+", without_parentheses))


def library_size(lib: List[Abstraction]) -> int:
    """
    count the constructors in the rhs (the body)
    ["fn_0(#0,#1) := (#1 (#0 T T))",
     "fn_1(#0,#1) := (fn_0 #1 #0 T)"]
    """
    return sum([program_size(program=ab.body) for ab in lib])


# abstraction learning and rewrite
def rewrite_with_abstractions(
    programs_test: List[str], programs_train: List[str], stitch_config: Dict
) -> Dict[str, Any]:
    """
    rewrite the a given set of programs with existing abstractions (library) learned from training programs and,
    return a dictionary containing summary stats/results of the rewrite operation.

    Returns:
        rewritten_programs : List[str]
        used_abstractions : List[Abstraction]
        rewritten_size : int
        used_library_size : int
        compressed_encoding_size: int (rewritten_size + used_library_size)
            --> the conditional/cross encoding size (C(test|train))
    """
    abs_from_train = compress(programs=programs_train, **stitch_config).abstractions

    rewritten_res = rewrite(
        programs=programs_test, abstractions=abs_from_train, **stitch_config
    )
    rewritten_programs = rewritten_res.rewritten
    used_abs = rewritten_res.json["abstractions"]
    rewritten_size = sum(program_size(x) for x in rewritten_programs)
    used_lib_size = sum(program_size(ab["body"]) for ab in used_abs)
    original_program_size = [program_size(x) for x in programs_test]
    num_abs = len(used_abs)
    res = {
        "rewritten_size": rewritten_size,
        "used_library_size": used_lib_size,
        "original_program_size": original_program_size,
        "number_of_used_abs": num_abs,
        "compressed_encoding_size": rewritten_size + used_lib_size,
        "rewritten_programs": rewritten_programs,
        "used_abstractions": used_abs,
    }
    return res


def get_global_library(corpus: Corpus, stitch_config: Dict) -> List[Abstraction]:
    """
    get the global library
    """
    return compress(programs=corpus.programs, **stitch_config).abstractions
