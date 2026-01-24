# QUANTT Pairs Trading

Pairs trading research sandbox focused on KO/PEP mean-reversion. The repo has two parallel tracks:
- **trading_part1** – a simple z-score spread strategy built with pandas/matplotlib.
- **trading_test** – a vectorbt-powered backtester for faster iteration and richer stats.

Method highlights:
- Hedge ratio from OLS: $y = \alpha + \beta x$ where $y$ = KO, $x$ = PEP; spread $S_t = KO_t - \beta\,PEP_t$.
- Z-score signals: enter when $|Z_t| > 2$, flatten when $|Z_t| < 0.5$ over a 30-day window.
- Performance metrics: total return, annualized Sharpe, max drawdown, trade count.

## Quickstart
1) Install Python 3.10+ and create a virtualenv.
2) Install deps:
	```bash
	pip install -r requirements.txt
	```
3) Get KO/PEP data (downloads via yfinance and saves to data/KO_PEP.csv):
	```bash
	python trading/trading_part1/data.py
	```
4) Run the simple pandas backtest (prints metrics and shows plots):
	```bash
	python trading/trading_part1/backtest.py
	```
5) Run the vectorbt version (prints headline stats):
	```bash
	python trading/trading_test/backtest2.py
	```

## Repo layout
- trading/trading_part1/data.py – downloads/prunes price data and saves CSV.
- trading/trading_part1/hedge_ratio.py – computes OLS hedge ratio and spread helpers.
- trading/trading_part1/signals.py – builds z-score signals from the spread.
- trading/trading_part1/backtest.py – pandas backtest with basic performance metrics and charts.
- trading/trading_test/backtest2.py – vectorbt backtesting harness with KO/PEP example.
- part2/cointegration, part2/identifying – placeholders for future Engle–Granger and clustering work.
- resources/ – background reading lists.

## How the simple strategy works
- Load daily closes for two tickers and drop NaNs.
- Fit OLS to get hedge ratio $\beta$; compute spread $S_t = KO_t - \beta\,PEP_t$.
- Compute rolling mean/std over a 30-day window to get z-score $Z_t = \frac{S_t - \mu_t}{\sigma_t}$.
- Signals: short spread when $Z_t > 2$; long spread when $Z_t < -2$; close when $|Z_t| < 0.5$.
- Backtest assumes dollar-neutral weights with equal and opposite exposure on both legs.

## Notes and tips
- File paths: backtest scripts currently point at windows-style sample paths; prefer the saved CSV at `trading/trading_part1/data/KO_PEP.csv` or switch to yfinance pulls (vectorbt flow supports either via `source='yfinance'`).
- Plots require a GUI backend; on headless machines set `matplotlib` to Agg or skip plotting sections.
- Vectorbt backtests may need `numba` install via pip wheels on macOS; ensure `brew install libomp` if compilation errors appear.

## Future improvements
- Flesh out `part2` Engle–Granger tests and clustering for candidate pair discovery.
- Add unit tests (pytest) around signal generation and portfolio stats.
- Parameter sweeps for z-score windows and thresholds; maybe grid search with joblib.
- Streamlit dashboard for interactive pair selection and live charts.

## License
Add a license file if you intend to share or publish.
