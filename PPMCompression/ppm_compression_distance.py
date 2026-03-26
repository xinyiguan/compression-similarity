import numpy as np
from collections.abc import Callable
from numpy.typing import ArrayLike, NDArray
from rpy2.robjects.packages import importr

ppm = importr("ppm")


def distance(
    x: ArrayLike,
    y: ArrayLike,
    alphabet_size: int | None = None,
    aggregation: Callable = np.mean,
) -> float:
    """
    Compression distance using Prediction by Partial Matching.
    Reimplimentation of IDyOM compression distance usied by Pearce & Mullensiefen (2017).

    :param x: A list of symbols (note: symbols must be accepted by np.searchsorted)
    :param y: Another list
    :param alphabet_size: Optional alphabet size; if not specified, combined alphabet of `x` and `y` used
    :param aggregation: Function used to combine event-level predictions across sequence
    """
    alphabet = np.unique(np.concatenate([x, y]))
    if alphabet_size and alphabet.size > alphabet_size:
        raise Exception("Alphabet size of x and y larger than specified size.")
    n = alphabet_size if alphabet_size else alphabet.size

    seq_x = as_iseq(x, alphabet)
    seq_y = as_iseq(y, alphabet)
    c_x = compress_seq(seq_x, None, n, aggregation)
    c_y = compress_seq(seq_y, None, n, aggregation)
    c_xy = compress_seq(seq_x, seq_y, n, aggregation)
    c_yx = compress_seq(seq_y, seq_x, n, aggregation)

    return max(c_xy, c_yx) / max(c_x, c_y)


def compress_seq(
    seq: NDArray[np.int_],
    pretrain_seq: NDArray[np.int_] | None,
    alphabet_size: int,
    aggregation: Callable,
) -> float:
    """
    Compress a single sequence.
    If pretraining sequence is given, first train on `pretrain_seq`, then predict `seq`.
    If no `pretrain_seq`, train and predict `seq`.
    """
    model = ppm.new_ppm_simple(alphabet_size)
    if pretrain_seq:
        ppm.model_seq(model, pretrain_seq, train=True, predict=False)
        train = False
    else:
        train = True
    res = ppm.model_seq(model, seq, train=train, predict=True)
    return aggregation(res.rx2("information_content"))


def as_iseq(x: ArrayLike, alphabet: ArrayLike) -> NDArray[np.int_]:
    """Return sequence of indicies (1-indexed, for R) for list of symbols, given an alphabet."""
    return (np.searchsorted(alphabet, x) + 1).tolist()
