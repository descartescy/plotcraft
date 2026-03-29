import math
import numpy as np
import pandas as pd
from fractions import Fraction

def floor_significant_digits(x:int | float, digits:int) -> int | float:
    """Round a number DOWN to the specified number of significant digits.

    This function always rounds toward negative infinity
    to retain a fixed number of significant digits,
    without rounding up. This is especially useful for
    truncating values strictly downward for numerical precision.

    Parameters
    ----------
    x : int or float
        Input number to be rounded down to significant digits.

    digits : int
        Number of significant digits to retain.
        Must be a positive integer.

    Returns
    -------
    int or float
        The input value rounded down to the specified
        number of significant digits.

    Raises
    ------
    ValueError
        If ``digits`` is not a positive integer.

    Examples
    --------
    >>> floor_significant_digits(123456, 2)
    120000
    >>> floor_significant_digits(-123456, 2)
    -120000
    >>> floor_significant_digits(1.23456, 2)
    1.2
    >>> floor_significant_digits(-1.23456, 2)
    -1.2
    """
    if digits <= 0 or type(digits) != int:
        raise ValueError("floor significant digits should be positive int")
    if x == 0:
        return 0
    elif x > 0:
        exp = math.floor(math.log10(x))
        decimals = exp - digits + 1
        if decimals < 0:
            decimals = -decimals
            scale = 10 ** decimals
            return math.floor(x * scale) / scale
        else:
            scale = 10 ** decimals
            return math.floor(x / scale) * scale
    else:
        x = abs(x)
        return -floor_significant_digits(x,digits)


def _sum_I(thresholds: np.ndarray, op: str, values: np.ndarray) -> np.ndarray:
    """
    For each threshold t in thresholds, count the number of elements in values that satisfy (t <= top values).

    op == ">"  →  count(t > v)  or count(v < t)
    op == "<"  →  count(t < v)  or count(v > t)
    """
    thresholds = np.asarray(thresholds, dtype=float)
    values = np.asarray(values, dtype=float)
    if op == ">":
        sv = np.sort(values)
        return np.searchsorted(sv, thresholds, side="left").astype(float)
    elif op == "<":
        sv = np.sort(values)
        return (len(values) - np.searchsorted(sv, thresholds, side="right")).astype(float)
    else:
        raise ValueError(f"op must be '>' or '<', got {op!r}")


def calculate_nb(
        real:np.ndarray,
        score:np.ndarray,
        thresholds: np.ndarray,
        casecontrol_rho: float | None = None,
        opt_in: bool = True,
) -> pd.DataFrame:
    N = len(real)
    d_pos = real == 1
    d_neg = real == 0
    n_pos = d_pos.sum()
    n_neg = d_neg.sum()
    tnf = _sum_I(thresholds, ">", score[d_neg]) / n_neg
    fnf = _sum_I(thresholds, ">", score[d_pos]) / n_pos
    tpf = _sum_I(thresholds, "<", score[d_pos]) / n_pos
    fpf = _sum_I(thresholds, "<", score[d_neg]) / n_neg
    if casecontrol_rho is None:
        rho = d_pos.mean()
        prob_high_risk = _sum_I(thresholds, "<", score) / N
    else:
        rho = casecontrol_rho
        prob_high_risk = (
                rho * _sum_I(thresholds, "<", score[d_pos]) / n_pos
                + (1 - rho) * _sum_I(thresholds, "<", score[d_neg]) / n_neg
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        if opt_in:
            odds = np.where(thresholds < 1.0, thresholds / (1 - thresholds), np.inf)
            nb = tpf * rho - odds * (1 - rho) * fpf
            snb = np.where(rho > 0, nb / rho, np.nan)
        else:
            inv_odds = np.where(thresholds > 0, (1 - thresholds) / thresholds, np.inf)
            nb = tnf * (1 - rho) - fnf * rho * inv_odds
            snb = np.where((1 - rho) > 0, nb / (1 - rho), np.nan)

    dp = tpf * rho
    non_dp = (1 - fpf) * (1 - rho)

    return pd.DataFrame({
        "threshold": thresholds,
        "FPR": fpf,
        "FNR": fnf,
        "TPR": tpf,
        "TNR": tnf,
        "NB": nb,
        "sNB": snb,
        "rho": rho,
        "prob_high_risk": prob_high_risk,
        "prob_low_risk": 1 - prob_high_risk,
        "DP": dp,
        "nonDP": non_dp,
    })

def _threshold_to_cost_benefit(thresholds: np.ndarray,
                                policy: str = "opt-in") -> list[str]:
    """
    将阈值转换为 cost:benefit 比值字符串，复现 R 中 fractions() 逻辑。
    opt-in  : cost/benefit = pt / (1 - pt)
    opt-out : cost/benefit = (1-pt) / pt
    """
    labels = []
    for pt in thresholds:
        if policy == "opt-in":
            num, den = pt, 1 - pt
        else:
            num, den = 1 - pt, pt
        if pt in (0.0, 1.0):
            labels.append("NA")
            continue
        frac = Fraction(num / den).limit_denominator(1000)
        labels.append(f"{frac.numerator}:{frac.denominator}")
    return labels