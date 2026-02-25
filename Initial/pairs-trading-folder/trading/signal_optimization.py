import itertools
import numpy as np
import pandas as pd

from pairs_trading.trading.signals import generate_signals


def _build_position_from_signal(signal: pd.Series) -> pd.Series:
    """
     Enter when signal becomes +/-1, hold that position until signal goes 0
    """
    pos = pd.Series(0, index=signal.index, dtype=int)
    current = 0
    for i in range(len(signal)):
        s = int(signal.iat[i])
        if s == 0:
            current = 0
        elif s in (-1, 1):
            current = s
        pos.iat[i] = current
    return pos


def _sharpe(pnl: pd.Series, annualization: float = 252) -> float:
    pnl = pnl.dropna()
    if len(pnl) < 2:
        return -np.inf
    mu = pnl.mean()
    sd = pnl.std(ddof=1)
    if sd <= 0 or not np.isfinite(sd):
        return -np.inf
    return float(mu / sd * np.sqrt(annualization))


def optimize_zscore_params(
    data: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    lookbacks=(20, 30, 60, 90, 120),
    entry_zs=(1.5, 2.0, 2.5),
    exit_zs=(0.0, 0.5, 1.0),
    min_trades: int = 15,
) -> pd.DataFrame:
    """
    Grid search for z-score parameters, returns results DataFrame sorted by Sharpe.
    """
    results = []

    for lookback, entry_z, exit_z in itertools.product(lookbacks, entry_zs, exit_zs):
        if exit_z >= entry_z:
            continue

        # Generate signals with current parameters
        sig_df = generate_signals(data, ticker1, ticker2, days=lookback)

        # New entry/exit logic
        z = sig_df["zscore"]
        raw = pd.Series(0, index=z.index, dtype=int)
        raw[z > entry_z] = -1
        raw[z < -entry_z] = 1
        raw[z.abs() < exit_z] = 0

        pos = _build_position_from_signal(raw)

        spread = sig_df["spread"]
        pnl = pos.shift(1).fillna(0) * spread.diff().fillna(0)

        # Count trades
        trades = int(((pos != 0) & (pos.shift(1).fillna(0) == 0)).sum())
        if trades < min_trades:
            continue

        s = _sharpe(pnl, annualization=252)

        results.append({
            "lookback": lookback,
            "entry_z": entry_z,
            "exit_z": exit_z,
            "trades": trades,
            "sharpe": s,
            "total_pnl": float(pnl.sum())
        })

    out = pd.DataFrame(results)
    if len(out) == 0:
        return out

    return out.sort_values(["sharpe", "trades"], ascending=[False, False]).reset_index(drop=True)
