# TransientCorrelation File Guide

Every file in this directory, what it does, and how files relate to each other.

---

## Root Files

| File | Purpose |
|------|---------|
| `config.py` | All thresholds and parameters as frozen dataclasses. Every module imports from here. Change a threshold once and it propagates everywhere. |
| `requirements.txt` | Python dependencies for this project specifically. |
| `pyproject.toml` | Python project metadata and pytest configuration. |
| `.gitignore` | Ignores `.venv`, `__pycache__`, `.DS_Store`, and the `research/data/` and `screener/data/` directories (pickle files are large). |
| `PROJECT_GUIDE.md` | Deep-dive explanation of the entire project: what it does, how the pipeline works, all results, key concepts. **Read this first.** |
| `FILE_GUIDE.md` | This file. What every file does. |

---

## validation/ — Core Statistical Testing Library

The lowest-level module. Everything else depends on it.

| File | Purpose |
|------|---------|
| `__init__.py` | Package init (empty). |
| `pair_validation.py` | Core functions used by every other module: `compute_hedge_ratio()` (OLS, TLS, Kalman), `half_life()`, `spread_cv_normalized()`, `zscore_signals()` (generates long/short/exit signals from z-scores), `simulate_spread_pnl()` (runs a simulated backtest on signals), `validate_pair()` (computes all metrics for a pair at a timestamp), `feature_shuffle_permutation_test()` (permutation testing for statistical significance). |

**Depends on:** numpy, pandas, scipy, statsmodels, pykalman
**Depended on by:** signals/, trading/, screener/

---

## signals/ — Feature Engineering and Clustering

Computes per-ticker features and runs the clustering pipeline.

| File | Purpose |
|------|---------|
| `__init__.py` | Package init (empty). |
| `features.py` | Computes 9 features per ticker per timestamp: returns, vol_short (50h), vol_medium (147h), beta_SPX_short, beta_SPX_medium, beta_sector_short, beta_sector_medium, RSI (70h), momentum_5H. Also computes 3 regime shift indicators (short vs medium window differences for vol, beta_SPX, beta_sector). |
| `detection.py` | Clustering pipeline: `run_clustering_snapshot()` runs StandardScaler -> PCA (90% variance) -> OPTICS on a single timestamp. `build_cluster_history()` runs across all timestamps. `compute_co_cluster_freq()` counts pair co-clustering. `detect_formation_events()` builds the full formation/dissolution timeline. `detect_new_formations()` compares consecutive snapshots. |
| `transient.py` | Transient event validation (backward-looking method validation). `validate_transient_event()` runs the 3-window approach on a single formation event: 2 obs execution lag, 20 obs calibration, 40 obs exploitation. Tests 5 criteria (correlation > 0.70, spread CV < 0.03, half-life < 8h, hedge drift < 0.20, has signal). `generate_transient_signals()` produces real-time z-score signals for a pair (infrastructure for future real-time trading). |
| `stable.py` | Tracks pairs that co-cluster consistently (stable pair identification and classification). |

**Depends on:** validation/, config.py
**Used by:** research/ notebooks

---

## trading/ — Backtesting and Trading Analysis

Pairs trading validation and backtesting for Phase 1.

| File | Purpose |
|------|---------|
| `__init__.py` | Package init (empty). |
| `trading.py` | `compute_noise_adjusted_frequency()` — the noise-adjusted co-cluster frequency calculation. `build_pair_registry()` — constructs the pair registry from OPTICS artifacts, filters by frequency threshold (>15%), adds consensus/significance flags. `test_pair_fundamentals()` — classical 3-criteria test (cointegration, half-life, Hurst) on calibration data. `backtest_pair()` — z-score mean-reversion backtest on a spread series. `walk_forward_backtest()` — 5 rolling cal/OOS splits for robust evaluation. `run_full_analysis()` — orchestrates testing + backtesting for all registry pairs. `load_artifacts()` — loads pickle files from research/data/. |

**Depends on:** validation/
**Used by:** research/ notebooks, screener/analysis.py (imports `compute_noise_adjusted_frequency`, `backtest_pair`, `get_daily_prices`, `hurst_exponent`)

---

## screener/ — Phase 2 Cross-Sector Analysis

Extended analysis for 142 tickers across 5 sectors.

| File | Purpose |
|------|---------|
| `__init__.py` | Package init (empty). |
| `screening.py` | Ticker screening via yfscreen library. |
| `universe.py` | 3-layer screening pipeline: liquidity ($2B+ market cap, 5M+ volume), sector classification, quality filtering. Constructs the 142-ticker universe. |
| `config.py` | Screener-specific configuration (sector lists, screening thresholds). Separate from root config.py. |
| `features_adapter.py` | Adapter that connects the signals/features.py feature computation to the screener's data format. |
| `analysis.py` | **5-test scored validation framework.** `validate_pair_relationship()` runs ADF, half-life, Hurst, variance ratio, and rolling correlation stability tests, returning a 0-5 score. `build_pair_registry()` constructs registry with 8% frequency threshold. `run_analysis()` orchestrates validation + backtesting for all registry pairs. `pair_type_summary()` and `sector_pair_breakdown()` compute summary statistics. `generate_report()` produces text report. |
| `enhanced_backtest.py` | **Enhanced backtesting.** `optimize_zscore_params()` grid-searches entry_z/exit_z/lookback on calibration data. `kalman_hedge_beta()` runs Kalman filter on calibration only, returns terminal beta (fixed for OOS — no look-ahead). `enhanced_backtest_pair()` compares 3 strategies: baseline (static z=2.0, OLS, no costs), enhanced (optimized z, OLS, 10bps costs), Kalman (Kalman terminal beta, optimized z, 10bps costs). |

**Depends on:** validation/, trading/, signals/
**Used by:** screener/notebooks/

---

## research/ — Phase 1 Semiconductor Notebooks

Proof-of-concept on 40 hand-picked semiconductor tickers. Notebooks are self-contained explorations that use the signals/, validation/, and trading/ modules.

| File | Purpose |
|------|---------|
| `optics-clustering.ipynb` | **Run first.** Downloads hourly price data via yfinance, computes features, runs OPTICS clustering across all timestamps, detects formation/dissolution events, computes co-cluster frequencies, classifies pairs. Saves all artifacts to `research/data/`. |
| `optics-signals.ipynb` | **Run second.** Loads OPTICS artifacts. Runs transient event validation (3-window approach) on formation events from registry pairs. Compares clustered vs random pair pass rates (4.0% vs 0.8% = 5x lift). Tests stable pairs with cointegration (0% pass rate). |
| `KMeans.ipynb` | Same pipeline as optics-clustering but using KMeans. Saves artifacts with `kmeans_` prefix. |
| `DBScan.ipynb` | Same pipeline using DBSCAN. Saves artifacts with `dbscan_` prefix. |
| `algorithm-comparison.ipynb` | **Run after all three algorithm notebooks.** Loads artifacts from all three, compares on common time range, identifies 8 consensus pairs, runs permutation tests, produces the algorithm ranking (OPTICS > KMeans > DBSCAN). |

### research/data/ — Pickle Artifacts (not in git)

These files are generated by the notebooks above. They must be present for notebooks to work without re-running.

| File Pattern | Content |
|-------------|---------|
| `ts_df.pkl` | Hourly price/feature DataFrame, MultiIndex (Datetime, Ticker). ~27 pickle files for OPTICS. |
| `cluster_history.pkl` | Per-timestamp cluster labels for every ticker. |
| `pair_co_cluster_freq.pkl` | Dict mapping (ticker_a, ticker_b) -> co-cluster count. |
| `df_pair_stability.pkl` | Pair frequency rankings. |
| `pair_classification.pkl` | Category labels (transient, stable, sporadic, unknown) per pair. |
| `df_formations.pkl` / `df_formations_actionable.pkl` | Formation event timelines. |
| `df_durations.pkl` | Formation/dissolution durations. |
| `oos_split_timestamp.pkl` | Train/test split point for OOS validation. |
| `kmeans_*.pkl` / `dbscan_*.pkl` | Same artifacts prefixed by algorithm name. |

---

## screener/notebooks/ — Phase 2 Analysis Pipeline

Run in order (01 through 05). Each notebook imports from the screener/, signals/, validation/, and trading/ modules.

| File | Purpose |
|------|---------|
| `01-screen-universe.ipynb` | Screens tickers via yfscreen, applies 3-layer filtering, downloads hourly prices, saves per-sector data to `screener/data/{sector}/`. |
| `02-clustering.ipynb` | Runs OPTICS clustering on each sector's data. Saves cluster_history, pair frequencies, formation events per sector. Also runs on combined multi-sector universe. |
| `03-signals-validation.ipynb` | Runs transient and stable pair validation per sector. Saves transient_results, stable_results per sector. |
| `04-algorithm-comparison.ipynb` | Multi-algorithm consensus analysis and permutation testing across sectors. Saves permutation_results per sector and combined. |
| `05-cross-sector-comparison.ipynb` | **Final analysis.** Runs 5-test scored validation on all registry pairs. Compares intra-sector vs cross-sector. Runs enhanced backtesting (baseline vs optimized vs Kalman). Produces final results tables. |

### screener/data/ — Pickle Artifacts (not in git)

Organized by sector. Each sector has its own subdirectory plus a `combined/` directory for cross-sector results.

| Directory | Content |
|-----------|---------|
| `technology/` | 14 pickle files: cluster_history, pair_co_cluster_freq, df_formations, df_durations, pair_registry, prices, tickers, ts_df, total_windows, consensus_pairs, permutation_results, sector_results, transient_results, stable_results |
| `healthcare/` | Same 14 files for healthcare tickers |
| `energy/` | Same 14 files for energy tickers |
| `financial_services/` | Same 14 files for financial services tickers |
| `industrials/` | Same 14 files for industrials tickers |
| `combined/` | 18 files: aggregated cross-sector results including analysis_results.pkl (5-test scores for all 3,643 pairs), enhanced_results.pkl (backtest results), walk_forward_results.pkl, plus all standard artifacts |
| Root files | `cross_sector_summary.pkl`, `cross_sector_report.txt`, `walk_forward_results.pkl` |

---

## tests/ — Test Suite

Run with `pytest tests/` from the TransientCorrelation directory.

| File | Purpose |
|------|---------|
| `__init__.py` | Package init (empty). |
| `test_validation.py` | Tests for validation/pair_validation.py: hedge ratio computation, half-life, spread metrics, z-score signals. |
| `test_signals.py` | Tests for signals/: feature computation, clustering snapshots, formation detection. |
| `test_trading.py` | Tests for trading/trading.py: noise-adjusted frequency, pair registry, backtesting. |
| `test_config.py` | Tests for config.py: verifies dataclass defaults and immutability. |
| `test_screener.py` | Tests for screener/analysis.py: 5-test validation, pair registry building. |

---

## Dependency Graph

```
config.py
    │
    ▼
validation/pair_validation.py    (core math: hedge ratios, z-scores, P&L simulation)
    │
    ├──► signals/features.py     (feature engineering)
    ├──► signals/detection.py    (clustering pipeline)
    ├──► signals/transient.py    (transient event validation)
    ├──► signals/stable.py       (stable pair tracking)
    │
    ├──► trading/trading.py      (Phase 1: registry, classical tests, backtesting)
    │
    └──► screener/analysis.py    (Phase 2: 5-test framework, registry)
         screener/enhanced_backtest.py  (Phase 2: Kalman, z-score optimization)
         screener/universe.py    (ticker screening)

research/notebooks ──► signals/, validation/, trading/, config.py
screener/notebooks ──► screener/, signals/, validation/, trading/, config.py
```

---

## Quick Reference: What to Read for What

| If you want to understand... | Read... |
|------------------------------|---------|
| The whole project | `PROJECT_GUIDE.md` |
| What every file does | This file (`FILE_GUIDE.md`) |
| How clustering works | `signals/detection.py` |
| How features are computed | `signals/features.py` |
| The 5-test validation framework | `screener/analysis.py` |
| The transient validation (method proof) | `signals/transient.py` |
| How backtesting works | `trading/trading.py` (basic), `screener/enhanced_backtest.py` (enhanced) |
| All thresholds and parameters | `config.py` |
| Phase 1 results | `research/algorithm-comparison.ipynb` |
| Phase 2 results | `screener/notebooks/05-cross-sector-comparison.ipynb` |
