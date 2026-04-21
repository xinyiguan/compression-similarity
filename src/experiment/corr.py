from typing import Literal, Optional

from numpy.typing import ArrayLike, Tuple
from scipy.stats import kendalltau, pearsonr, spearmanrho


def corr(a: ArrayLike, b: ArrayLike, 
               method:Literal["pearson", "spearman", "kendall"])->Tuple[float,float]:
    """
    Returns:
            - stats (r or tau)
            - pvalue
    """
    if method=="pearson":
        res = pearsonr(x=a, y=b)
    elif method=="spearman":
        res = spearmanrho(a=a, b=b)
    elif method=="kendall":
        res = kendalltau(x=a,y=b)
    else:
        raise ValueError
    
    stats = res.statistic
    p = res.pvalue
    
    return stats, p


def run(result_path: str, corr_method:Literal["pearson", "spearman", "kendall"]):
    """
    The result json file must contain the following fields (data):
    D_stitch, D_ppm.
    """
    
