
from dataclasses import dataclass
from typing import Callable, Dict, List, Protocol
import model.stitchCompression.runStitch as runStitch

# a function of type (a,a) -> float
type SimMeasure[a]= Callable[[a,a],float]

class HasSimilarity[a](Protocol):
    @classmethod
    def similarity(x:a,y:a) -> float: ...

# this corresponds to C(x|y)
type ConditionedOn[a] = Callable[[a,a],float]

def compression_distance[a](
    c : Callable[[a,a],float], # the C in C(x|y)
    ) -> Callable[[a,a],float]:
    return lambda x,y: max(c(x,y),c(y,x))/max(c(x,x),c(y,y))

def compression_distance_stitch(stitch_config:Dict) -> Callable[
    [List[str],List[str]],
    float
    ]:
    def f(x:List[str],y:List[str]): return runStitch.compressed_encoding_size(
        programs_test=x, 
        programs_train=y, 
        stitch_config=stitch_config)
    return compression_distance(f)

def corpusRun[a : HasSimilarity](xs:List[a],):
    for x in xs:
        a.similarity(x,x)
    