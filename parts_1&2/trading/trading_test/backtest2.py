"""VectorBT-powered backtesting helpers for pairs strategies.

This module wraps vectorbt so we can plug any signal generator or target
weight DataFrame and let vectorbt simulate the portfolio. Default wiring uses
the existing ``trading_part1.signals.generate_signals`` function to create
z-score based long/short spread trades.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt
from vectorbt.portfolio.enums import SizeType
import yfinance as yf

ROOT_DIR = Path(__file__).resolve().parents[1]
PART1_DIR = ROOT_DIR / "trading_part1"
for path in (ROOT_DIR, PART1_DIR):
    if str(path) not in sys.path:
        sys.path.append(str(path))

try:
    import signals as default_signals
except ModuleNotFoundError:
    default_signals = None


def load_prices(
    ticker1: str,
    ticker2: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: str = "yfinance",
    csv_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Load price data from yfinance or a CSV file."""

    if source == "yfinance":
        raw = yf.download(
            [ticker1, ticker2],
            start=start,
            end=end,
            progress=False,
            auto_adjust=False,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            level0 = set(raw.columns.get_level_values(0))
            level1 = set(raw.columns.get_level_values(1))
            pick = None
            pick_level = None
            for candidate in ("Adj Close", "Close"):
                if candidate in level1:
                    pick = candidate
                    pick_level = 1
                    break
                if candidate in level0:
                    pick = candidate
                    pick_level = 0
                    break

            if pick is None:
                raise KeyError(
                    "No close-like price level found; level0="
                    f"{sorted(level0)}, level1={sorted(level1)}"
                )

            data = raw.xs(pick, level=pick_level, axis=1)
        else:
            # Already adjusted/flattened; assume columns are tickers
            data = raw

        data = data[[ticker1, ticker2]].dropna()
        return data

    if source == "csv":
        candidate_paths = []
        if csv_path is not None:
            candidate_paths.append(Path(csv_path))
        else:
            candidate_paths.append(ROOT_DIR / "trading_part1" / "data" / f"{ticker1}_{ticker2}.csv")
            candidate_paths.append(ROOT_DIR / "data" / f"{ticker1}_{ticker2}.csv")

        for path in candidate_paths:
            if path.exists():
                data = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
                data = data[[ticker1, ticker2]].dropna()
                return data

        tried = " | ".join(str(p) for p in candidate_paths)
        raise FileNotFoundError(f"CSV not found. Tried: {tried}. Pass csv_path or set source='yfinance'.")

    raise ValueError("source must be 'yfinance' or 'csv'")


def default_signal_generator(
    prices: pd.DataFrame,
    ticker1: str,
    ticker2: str,
    signal_days: int = 30,
) -> pd.DataFrame:
    """Use the legacy signal generator if it is available."""

    if default_signals is None:
        raise ImportError("trading_part1.signals is not available on sys.path")
    return default_signals.generate_signals(prices, ticker1, ticker2, days=signal_days)


def weights_from_signal(
    signal: pd.Series,
    ticker1: str,
    ticker2: str,
    gross_leverage: float = 1.0,
) -> pd.DataFrame:
    """Convert +1/-1/0 signals into target weights for both legs."""

    clean = signal.ffill().fillna(0).astype(int)
    half_weight = 0.5 * gross_leverage
    w1 = np.where(clean == 1, half_weight, np.where(clean == -1, -half_weight, 0.0))
    w2 = -w1
    return pd.DataFrame({ticker1: w1, ticker2: w2}, index=clean.index)


def run_with_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    init_cash: float = 100_000.0,
    fees: float = 0.0005,
    slippage: float = 0.0005,
    freq: Optional[str] = None,
) -> vbt.Portfolio:
    """Run a vectorbt portfolio using target percent weights."""

    freq = freq or pd.infer_freq(prices.index) or "1D"
    aligned_weights = weights.reindex(prices.index).fillna(0.0)
    portfolio = vbt.Portfolio.from_orders(
        close=prices,
        size=aligned_weights,
        size_type=SizeType.TargetPercent,
        cash_sharing=True,
        fees=fees,
        slippage=slippage,
        init_cash=init_cash,
        freq=freq,
        group_by=True,
        call_seq="auto",
    )
    return portfolio


def summarize_portfolio(portfolio: vbt.Portfolio) -> Dict[str, float]:
    """Small set of headline stats from vectorbt outputs."""

    def first_scalar(val: object) -> float:
        if np.isscalar(val):
            return float(val)
        try:
            return float(val.iloc[0])  # type: ignore[arg-type]
        except Exception:
            return float(val)

    total_return = first_scalar(portfolio.total_return(group_by=True)) * 100
    sharpe = first_scalar(portfolio.sharpe_ratio(group_by=True))
    max_dd = first_scalar(portfolio.max_drawdown(group_by=True)) * 100
    trades_val = portfolio.trades.count(group_by=True)
    trades = int(first_scalar(trades_val))

    return {
        "total_return_pct": total_return,
        "annualized_sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "trades": trades,
    }


def backtest_pairs(
    ticker1: str,
    ticker2: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    source: str = "yfinance",
    csv_path: Optional[Path] = None,
    signal_func: Optional[Callable[..., pd.DataFrame]] = None,
    signal_kwargs: Optional[Dict[str, object]] = None,
    init_cash: float = 100_000.0,
    fees: float = 0.0005,
    slippage: float = 0.0005,
    gross_leverage: float = 1.0,
    freq: Optional[str] = None,
) -> Tuple[vbt.Portfolio, Dict[str, float]]:
    """End-to-end pairs backtest using vectorbt."""

    prices = load_prices(ticker1, ticker2, start=start, end=end, source=source, csv_path=csv_path)
    signal_func = signal_func or default_signal_generator
    signal_kwargs = signal_kwargs or {}
    signal_df = signal_func(prices, ticker1, ticker2, **signal_kwargs)
    if "signal" not in signal_df.columns:
        raise ValueError("signal_func must return a DataFrame with a 'signal' column")
    weights = weights_from_signal(signal_df["signal"], ticker1, ticker2, gross_leverage=gross_leverage)
    portfolio = run_with_weights(
        prices=prices[[ticker1, ticker2]],
        weights=weights,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )
    stats = summarize_portfolio(portfolio)
    return portfolio, stats


def backtest_with_custom_weights(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    init_cash: float = 100_000.0,
    fees: float = 0.0005,
    slippage: float = 0.0005,
    freq: Optional[str] = None,
) -> Tuple[vbt.Portfolio, Dict[str, float]]:
    """Backtest any strategy that provides target percent weights per asset."""

    portfolio = run_with_weights(
        prices=prices,
        weights=weights,
        init_cash=init_cash,
        fees=fees,
        slippage=slippage,
        freq=freq,
    )
    stats = summarize_portfolio(portfolio)
    return portfolio, stats


def example() -> None:
    """Quick KO/PEP demo run."""

    pf, stats = backtest_pairs(
        ticker1="KO",
        ticker2="PEP",
        start="2018-01-01",
        end="2024-12-31",
        signal_kwargs={"signal_days": 30},
    )
    print("Headline stats:")
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}")
    # Plotting requires optional front-end deps; omit in CLI example to avoid import errors.
    # pf.plot().show()


if __name__ == "__main__":
    example()