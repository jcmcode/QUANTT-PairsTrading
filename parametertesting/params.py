"""Parameter sweep for pairs strategy (z-score thresholds, windows, etc.).

This script runs a grid search over signal parameters and basic execution
settings, evaluates with a vectorbt backtester, and reports the best
configuration by total return (with Sharpe as a tie-breaker).

It does not modify any other files. Defaults target KO/PEP CSV included
under trading_part1/data for speed, but can also pull from yfinance.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

# Bring repo modules into path for reuse of existing helpers
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
	sys.path.append(str(ROOT_DIR))

try:
	from trading_test.backtest2 import (
		backtest_with_custom_weights,
		load_prices,
		summarize_portfolio,
		weights_from_signal,
	)
except Exception as e:  # pragma: no cover
	raise RuntimeError(
		"Failed to import trading_test.backtest2 helpers. Ensure workspace layout is unchanged."
	) from e


# -----------------------------
# Signal generation (local)
# -----------------------------

def _compute_hedge_ratio_ols(prices: pd.DataFrame, ticker1: str, ticker2: str) -> float:
	"""Estimate hedge ratio beta via OLS: ticker1 ~ const + beta * ticker2.

	Avoids relying on hardcoded-ticker helper; works for any pair.
	"""

	y = prices[ticker1].astype(float)
	X = sm.add_constant(prices[ticker2].astype(float))
	model = sm.OLS(y, X, missing="drop").fit()
	beta = float(model.params.get(ticker2))
	return beta


def generate_param_signal(
	prices: pd.DataFrame,
	ticker1: str,
	ticker2: str,
	*,
	signal_days: int = 30,
	entry_z: float = 2.0,
	exit_z: float = 0.5,
) -> pd.DataFrame:
	"""Build spread z-score and +1/0/-1 signals with custom thresholds.

	-1 = Spread high (short spread)  -> short ticker1, long ticker2
	+1 = Spread low  (long spread)   -> long ticker1, short ticker2
	 0 = Exit
	"""

	prices = prices[[ticker1, ticker2]].copy()
	beta = _compute_hedge_ratio_ols(prices, ticker1, ticker2)
	spread = prices[ticker1] - beta * prices[ticker2]

	mean = spread.rolling(window=signal_days, min_periods=signal_days).mean()
	std = spread.rolling(window=signal_days, min_periods=signal_days).std()
	z = (spread - mean) / std

	signal = pd.Series(0, index=prices.index, dtype=int)
	signal[z > entry_z] = -1
	signal[z < -entry_z] = 1
	signal[z.abs() < exit_z] = 0

	return pd.DataFrame({
		"spread": spread,
		"zscore": z,
		"signal": signal,
	})


# -----------------------------
# Grid search runner
# -----------------------------

@dataclass(frozen=True)
class ParamConfig:
	signal_days: int
	entry_z: float
	exit_z: float
	gross_leverage: float
	fees: float
	slippage: float

	def as_dict(self) -> Dict[str, float]:
		return {
			"signal_days": self.signal_days,
			"entry_z": self.entry_z,
			"exit_z": self.exit_z,
			"gross_leverage": self.gross_leverage,
			"fees": self.fees,
			"slippage": self.slippage,
		}


def _param_grid(
	days: Iterable[int],
	entries: Iterable[float],
	exits: Iterable[float],
	leverages: Iterable[float],
	fees: Iterable[float],
	slippages: Iterable[float],
) -> List[ParamConfig]:
	return [
		ParamConfig(d, e, x, g, f, s)
		for d, e, x, g, f, s in product(days, entries, exits, leverages, fees, slippages)
	]


def run_grid_search(
	prices: pd.DataFrame,
	ticker1: str,
	ticker2: str,
	params: List[ParamConfig],
	*,
	init_cash: float = 100_000.0,
	freq: Optional[str] = None,
) -> pd.DataFrame:
	"""Evaluate all param sets and return a results DataFrame sorted by total return."""

	rows: List[Dict[str, float]] = []

	for cfg in params:
		sig = generate_param_signal(
			prices,
			ticker1,
			ticker2,
			signal_days=cfg.signal_days,
			entry_z=cfg.entry_z,
			exit_z=cfg.exit_z,
		)
		w = weights_from_signal(sig["signal"], ticker1, ticker2, gross_leverage=cfg.gross_leverage)
		pf, stats = backtest_with_custom_weights(
			prices=prices[[ticker1, ticker2]],
			weights=w,
			init_cash=init_cash,
			fees=cfg.fees,
			slippage=cfg.slippage,
			freq=freq,
		)

		row: Dict[str, float] = {
			**cfg.as_dict(),
			**stats,
		}
		rows.append(row)

	results = pd.DataFrame(rows)
	# Sort by total return desc, then Sharpe desc
	results = results.sort_values(by=["total_return_pct", "annualized_sharpe"], ascending=[False, False])
	return results.reset_index(drop=True)


# -----------------------------
# CLI
# -----------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(description="Grid search pairs parameters (z-score, window, etc.)")
	p.add_argument("--ticker1", default="KO")
	p.add_argument("--ticker2", default="PEP")
	p.add_argument("--source", choices=["csv", "yfinance"], default="csv", help="Data source")
	p.add_argument("--csv-path", type=str, default=None, help="Optional CSV path override")
	p.add_argument("--start", type=str, default=None)
	p.add_argument("--end", type=str, default=None)
	p.add_argument("--init-cash", type=float, default=100_000.0)

	# Grid dimensions
	p.add_argument("--days", type=int, nargs="+", default=[20, 30, 60])
	p.add_argument("--entry", type=float, nargs="+", default=[1.5, 2.0, 2.5])
	p.add_argument("--exit", type=float, nargs="+", default=[0.3, 0.5, 0.7])
	p.add_argument("--gross", type=float, nargs="+", default=[1.0])
	p.add_argument("--fees", type=float, nargs="+", default=[0.0005])
	p.add_argument("--slippage", type=float, nargs="+", default=[0.0005])

	# Output controls
	p.add_argument("--sort-by", default="total_return_pct", choices=[
		"total_return_pct",
		"annualized_sharpe",
		"max_drawdown_pct",
		"trades",
	])
	p.add_argument("--top", type=int, default=10)
	p.add_argument("--save", type=str, default=None, help="Optional results CSV path; default under parametertesting/results-*.csv")
	p.add_argument("--no-save", action="store_true")
	p.add_argument("--quiet", action="store_true")
	return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
	args = parse_args(argv)

	# Load prices once for speed
	csv_path = Path(args.csv_path) if args.csv_path else None
	prices = load_prices(
		args.ticker1,
		args.ticker2,
		start=args.start,
		end=args.end,
		source=args.source,
		csv_path=csv_path,
	)

	grid = _param_grid(
		days=args.days,
		entries=args.entry,
		exits=args.exit,
		leverages=args.gross,
		fees=args.fees,
		slippages=args.slippage,
	)

	if not args.quiet:
		print(f"Running {len(grid)} configurations on {args.ticker1}/{args.ticker2} ({len(prices)} bars)...")

	results = run_grid_search(
		prices=prices,
		ticker1=args.ticker1,
		ticker2=args.ticker2,
		params=grid,
		init_cash=args.init_cash,
	)

	# Sort by user selection for display
	results = results.sort_values(by=[args.sort_by, "annualized_sharpe"], ascending=[False, False]).reset_index(drop=True)

	# Print top
	topn = min(args.top, len(results))
	display_cols = [
		"signal_days",
		"entry_z",
		"exit_z",
		"gross_leverage",
		"fees",
		"slippage",
		"total_return_pct",
		"annualized_sharpe",
		"max_drawdown_pct",
		"trades",
	]
	if not args.quiet:
		print("\nTop results:")
		print(results[display_cols].head(topn).to_string(index=False, justify="center", float_format=lambda x: f"{x:,.4f}"))

	# Save
	if not args.no_save:
		out_path = Path(args.save) if args.save else (ROOT_DIR / "parametertesting" / f"results_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv")
		out_path.parent.mkdir(parents=True, exist_ok=True)
		results.to_csv(out_path, index=False)
		if not args.quiet:
			print(f"\nSaved full results to {out_path}")

	# Brief winner summary
	best = results.iloc[0].to_dict()
	if not args.quiet:
		print("\nBest by total return:")
		print({k: best[k] for k in ["signal_days", "entry_z", "exit_z", "gross_leverage", "total_return_pct", "annualized_sharpe", "max_drawdown_pct", "trades"]})

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

