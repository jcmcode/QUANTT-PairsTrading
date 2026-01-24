QUANTT Pairs Trading Pipeline
=============================

End-to-end pairs trading sandbox: ingest price data, discover candidate pairs with clustering (K-Means, DBSCAN), validate with cointegration tests, and backtest a mean-reversion spread strategy.

Pipeline
- Ingest: pull and store adjusted closes.
- Identify: cluster assets (K-Means/DBSCAN) to surface likely pairs.
- Validate: run ADF / Engle–Granger cointegration tests on spreads.
- Trade: generate z-score signals, size legs dollar-neutrally, backtest.








