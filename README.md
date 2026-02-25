QUANTT Pairs Trading Pipeline
=============================

## Repository Structure

### `Initial/`
Original pairs trading work: data ingestion, pair identification (K-Means, DBSCAN), cointegration validation (ADF, Engle-Granger), and backtesting with a Streamlit frontend. Self-contained.

### `TransientCorrelation/`
Two-phase transient correlation detection research. Uses unsupervised clustering to discover short-term relationships between assets, validates the method, then trades persistently co-clustering pairs.

- **Phase 1 (Semiconductors):** 40 tickers, 3 algorithms (OPTICS, DBSCAN, KMeans), 8 consensus pairs, 5x lift over random
- **Phase 2 (Cross-Sector):** 142 tickers across 5 sectors, 5-test scored validation, enhanced backtesting with Kalman hedge ratios

See `TransientCorrelation/PROJECT_GUIDE.md` for the full deep-dive, or `TransientCorrelation/FILE_GUIDE.md` for what every file does.

```bash
cd TransientCorrelation
pip install -r requirements.txt
pytest tests/
jupyter notebook
```





