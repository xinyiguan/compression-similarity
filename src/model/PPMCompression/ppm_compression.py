import numpy as np
from collections.abc import Callable
from ppm import PPMSimple


def compress(
    a: list,
    b: list | None = None,
    pretrain_a: bool = True,
    alphabet_size: int | None = None,
    aggregation: Callable = np.mean,
    ppm_params: dict = {},
) -> float:
    """
    Compress a sequence using PPM.
    If ``pretrain_a`` is True, first train model on ``a``, then predict.
    If sequence b is also given, first train on ``b``, then predict ``a``.
    """
    if not alphabet_size and not b:
        alphabet_size = len(set(a))
    elif not alphabet_size and b:
        alphabet_size = len(set(a + b))

    model = PPMSimple(alphabet_size, **ppm_params)
    if pretrain_a:
        model.model_seq(a, train=True, predict=False)
    if b:
        model.model_seq(b, train=True, predict=False)
    result = model.model_seq(a, train=False, predict=True)

    return float(aggregation(result.information_content))
