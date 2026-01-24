QUANTT Pairs Trading Pipeline
=============================

End-to-end pairs trading sandbox: ingest price data, discover candidate pairs with clustering (K-Means, DBSCAN), validate with cointegration tests, and backtest a mean-reversion spread strategy.

Pipeline
- Ingest: pull and store adjusted closes.
- Identify: cluster assets (K-Means/DBSCAN) to surface likely pairs.
- Validate: run ADF / Engle–Granger cointegration tests on spreads.
- Trade: generate z-score signals, size legs dollar-neutrally, backtest.

Setup
- Python 3.10+ recommended.
- Install deps: `pip install -r requirements.txt`.
- Core libs: pandas, numpy, statsmodels, scikit-learn, yfinance, matplotlib, vectorbt.

Project layout
- [trading/trading_part1/data.py](trading/trading_part1/data.py) – download KO/PEP prices via yfinance and save to [trading/trading_part1/data/KO_PEP.csv](trading/trading_part1/data/KO_PEP.csv).
- [trading/trading_part1/hedge_ratio.py](trading/trading_part1/hedge_ratio.py) – OLS hedge ratio, spread helpers, rolling stats.
- [trading/trading_part1/signals.py](trading/trading_part1/signals.py) – z-score signal engine.
- [trading/trading_part1/backtest.py](trading/trading_part1/backtest.py) – pandas backtest with basic performance metrics and plots.
- [trading/trading_test/backtest2.py](trading/trading_test/backtest2.py) – vectorbt backtester; flexible weights, yfinance or CSV sourcing, headline stats.
- [part2/identifying/k_means.py](part2/identifying/k_means.py), [part2/identifying/dbscan.py](part2/identifying/dbscan.py) – clustering stubs for pair discovery.
- [part2/cointegration/adf.py](part2/cointegration/adf.py), [part2/cointegration/engle_granger.py](part2/cointegration/engle_granger.py) – cointegration testing stubs.
- [resources/](resources) – reading lists.

How the trading leg works
- Hedge ratio via OLS: $$KO_t = \alpha + \beta\,PEP_t + \varepsilon_t$$
- Spread: $$S_t = KO_t - \beta\,PEP_t$$
- Rolling z-score over window $w$ (default 30): $$Z_t = \frac{S_t - \mu_t}{\sigma_t}$$
- Signals: enter short spread when $Z_t > 2$, long when $Z_t < -2$, flatten when $|Z_t| < 0.5$.
- Positioning: dollar-neutral, weights ±0.5 per leg by default in vectorbt path.

Quickstart
1) Pull data and save CSV:
	- `python trading/trading_part1/data.py`
2) Run simple pandas backtest (prints metrics, shows plots):
	- `python trading/trading_part1/backtest.py`
3) Run vectorbt backtest (prints headline stats):
	- `python trading/trading_test/backtest2.py`

Planned extensions
- Implement K-Means/DBSCAN clustering to surface candidate pairs from universes.
- Add Engle–Granger / ADF wrappers to score spreads before trading.
- Parameter sweeps for z-score windows/thresholds; add pytest coverage around signals and stats.
- Streamlit dashboard for interactive pair selection and live visualization.

Notes
- Some part2 modules are placeholders; fill them when expanding pair discovery and cointegration validation.
- matplotlib plots need a GUI backend; use Agg or skip plotting in headless runs.
- On macOS, vectorbt/numba may require `brew install libomp` before `pip install -r requirements.txt`.
