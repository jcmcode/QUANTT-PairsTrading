import pandas as pd
import statsmodels.tsa.stattools as stat


def align_series(y0, y1):
    df = pd.concat([y0.rename("y0"), y1.rename("y1")], axis=1).dropna()
    return df["y0"], df["y1"]


def engle_granger_test(y0, y1, trend="c", maxlag=None, autolag="aic"):
    y0_aligned, y1_aligned = align_series(y0, y1)

    coint_t, pvalue, crit_values = stat.coint(
        y0_aligned.values,
        y1_aligned.values,
        trend=trend,
        method="aeg",
        maxlag=maxlag,
        autolag=autolag,
        return_results=None,
    )

    return float(coint_t), float(pvalue), [float(v) for v in crit_values]