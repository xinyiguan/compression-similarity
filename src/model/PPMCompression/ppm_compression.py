import numpy as np
from collections.abc import Callable
from itertools import chain
from ppm import PPMSimple


def compress(
    target_seqs: list[list],
    training_seqs: list[list] = [],
    alphabet_size: int | None = None,
    aggregation: Callable = sum,
    ppm_params: dict = {},
) -> float:
    """
    Compress sequences using PPM.
    Model pretrained on all sequences in ``training_seqs``.
    Model predicts all sequences in ``target_seqs``, without training.
    Returns aggregated information content across all events in seqs.
    """
    for seq in target_seqs:
        assert isinstance(seq, list), "sequence must be a list"
    for seq in training_seqs:
        assert isinstance(seq, list), "sequence must be a list"

    if not alphabet_size:
        alphabet_size = len(set(chain(*target_seqs, *training_seqs)))

    model = PPMSimple(alphabet_size, **ppm_params)
    for seq in training_seqs:
        model.model_seq(seq, train=True, predict=False)
    result = []
    for seq in target_seqs:
        output = model.model_seq(seq, train=False, predict=True)
        result += output.information_content.tolist()

    return float(aggregation(result))
