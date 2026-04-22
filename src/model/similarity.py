
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
    c : Callable[[a,a],float] # the C in C(x|y)
    ) -> Callable[[a,a],float]:
    """
    ncd = ( max( C(x|y), C(y|x)) / max ( C(x), C(y) ) )
    """
    return lambda x,y: max(c(x,y),c(y,x))/max(c(x,x),c(y,y))

def compression_distance2[a](
    c_xy: Callable[[a,a],float], # the C in C(xy) contactenation
    c : Callable[[a], float]
) -> Callable[[a,a],float]:
    """
    ncd = ( C(xy) - min( C(x), C(y) ) ) / max( C(x), C(y) )
    """
    return lambda x,y: (c_xy(x,y)-min(c(x), c(y)))/max(c(x), c(y))

def corpusRun[a : HasSimilarity](xs:List[a],):
    for x in xs:
        a.similarity(x,x)
    