================================================================================
PAIRS TRADING PIPELINE - PROJECT STRUCTURE & ARCHITECTURE
================================================================================

PROJECT OVERVIEW
================================================================================

This project implements an end-to-end pairs trading strategy discovery and 
backtesting system. The codebase is organized into three main layers:

1. BACKEND (pairs-trading/) - Core algorithmic logic
2. FRONTEND (frontend/) - Streamlit visualization dashboard
3. DATA (data/) - Raw and processed data storage

The pipeline flows: Data Loading → Feature Preprocessing → Clustering → 
Cointegration Testing → Backtesting → Performance Metrics

================================================================================
FOLDER STRUCTURE (DETAILED)
================================================================================

pairs-trading/
│
├── __init__.py
│   Marks this directory as a Python package. Allows imports like:
│   from pairs_trading import pipeline, data, trading, etc.
│
├── main.py
│   Main entry point for the entire backend.
│   Usage: python pairs-trading/main.py
│   Responsibilities:
│   - Parse command line arguments (tickers, date ranges, parameters)
│   - Call pipeline orchestrator with user inputs
│   - Return/save final results
│
├── pipeline.py ⭐ MAIN ORCHESTRATOR
│   High-level coordination of the entire workflow.
│   Responsibilities:
│   - Accept inputs (list of tickers, date range, algorithm parameters)
│   - Call data loading → preprocessing → clustering → validation → backtesting
│   - Aggregate results and metrics
│   - Return structured output (dict/object with all results)
│   
│   Key Functions:
│   - run_full_pipeline(tickers, start_date, end_date, **kwargs)
│     Returns: {
│         'prices': DataFrame,
│         'features': DataFrame,
│         'clusters': {kmeans: [...], dbscan: [...]},
│         'valid_pairs': List[Tuple[str, str]],
│         'backtest_results': {pair: {equity, trades, metrics}},
│         'summary_metrics': DataFrame
│     }
│
├── data/
│   │
│   ├── __init__.py
│   │
│   ├── loader.py
│   │   Responsibility: Load price data from various sources
│   │
│   │   Key Functions:
│   │   - load_prices(tickers: List[str], start_date, end_date, source='yfinance') → DataFrame
│   │     * Loads adjusted close prices for given tickers
│   │     * Returns: DataFrame(index=dates, columns=tickers, values=prices)
│   │     * Handles missing data, aligns dates across tickers
│   │
│   │   - load_prices_from_csv(file_path) → DataFrame
│   │     * Alternative: Load from existing CSV (e.g., data/raw/KO_PEP.csv)
│   │
│   │   Example Usage:
│   │   prices = data.loader.load_prices(['KO', 'PEP'], '2020-01-01', '2023-12-31')
│   │
│   └── preprocessor.py
│       Responsibility: Transform raw prices into features for clustering
│
│       Key Functions:
│       - calculate_returns(prices: DataFrame) → DataFrame
│         * Daily/log returns for each ticker
│         * Returns: DataFrame with same shape as prices
│
│       - normalize_features(data: DataFrame) → DataFrame
│         * Standardize each column (mean=0, std=1)
│         * Important for clustering algorithms
│
│       - reduce_dimensions(features: DataFrame, n_components=2) → DataFrame
│         * PCA dimensionality reduction to 2D for visualization
│         * Returns: DataFrame(shape: n_tickers x 2) for scatter plotting
│
│       - prepare_features(prices: DataFrame, method='returns') → DataFrame
│         * Orchestrator: handles returns calculation, normalization, etc.
│         * Returns: Ready-to-use feature matrix for clustering
│
│       Example Usage:
│       returns = data.preprocessor.calculate_returns(prices)
│       features = data.preprocessor.normalize_features(returns)
│       features_2d = data.preprocessor.reduce_dimensions(features)
│
├── identification/
│   │
│   ├── __init__.py
│   │
│   ├── kmeans.py
│   │   Responsibility: K-Means clustering algorithm
│   │
│   │   Key Functions:
│   │   - fit_kmeans(features: DataFrame, n_clusters: int, random_state=42) → KMeansResult
│   │     * Input: Features (normalized, typically returns or normalized prices)
│   │     * Output: Object with:
│   │       - labels: array of cluster assignments for each ticker
│   │       - centers: cluster centroids
│   │       - inertia: within-cluster sum of squares
│   │     * Selects k number of clusters (user parameter, range 2-10)
│   │
│   │   Example Usage:
│   │   result = identification.kmeans.fit_kmeans(features, n_clusters=3)
│   │   clusters = result.labels  # [0, 1, 0, 2, 1, ...] for each ticker
│   │
│   └── dbscan.py
│       Responsibility: DBSCAN clustering algorithm (density-based)
│
│       Key Functions:
│       - fit_dbscan(features: DataFrame, eps: float, min_samples: int) → DBScanResult
│         * Input: Features (normalized)
│         * Parameters:
│           - eps: neighborhood radius (typical range 0.1-2.0)
│           - min_samples: min points in neighborhood for core point (typical 2-10)
│         * Output: Object with:
│           - labels: cluster assignments (-1 for noise/outliers)
│           - n_clusters: number of clusters found
│         * Advantage: finds natural cluster count, handles outliers
│
│       Example Usage:
│       result = identification.dbscan.fit_dbscan(features, eps=0.5, min_samples=3)
│
│   --- After clustering, pairs are extracted as all combinations within each cluster ---
│   Example: If cluster_0 = ['AAPL', 'MSFT', 'GOOG'], then pairs are:
│   ('AAPL', 'MSFT'), ('AAPL', 'GOOG'), ('MSFT', 'GOOG')
│
├── validation/
│   │
│   ├── __init__.py
│   │
│   ├── adf.py
│   │   Responsibility: Augmented Dickey-Fuller stationarity test
│   │   Used to check if individual time series are stationary
│   │
│   │   Key Functions:
│   │   - adf_test(series: pd.Series, verbose=False) → ADFResult
│   │     * Input: Time series (price or spread)
│   │     * Output: Object with:
│   │       - test_statistic: ADF test statistic
│   │       - p_value: p-value for hypothesis test
│   │       - critical_values: dict of critical values (1%, 5%, 10%)
│   │       - is_stationary(alpha=0.05): bool (p_value < alpha)
│   │
│   │   Example Usage:
│   │   result = validation.adf.adf_test(prices['KO'])
│   │   if result.is_stationary():  # If p-value < 0.05
│   │       print("Series is stationary")
│   │
│   └── engle_granger.py
│       Responsibility: Engle-Granger cointegration test
│       Used to validate that pairs move together long-term
│
│       Key Functions:
│       - engle_granger_test(series1: pd.Series, series2: pd.Series) → EGResult
│         * Input: Two price time series
│         * Output: Object with:
│           - test_statistic: EG test statistic
│           - p_value: p-value for cointegration test
│           - hedge_ratio: optimal weight for spread = price1 - hedge_ratio * price2
│           - is_cointegrated(alpha=0.05): bool (p_value < alpha)
│           - spread: calculated spread time series using hedge_ratio
│
│       - test_pairs(candidate_pairs: List[Tuple], prices: DataFrame, alpha=0.05) → DataFrame
│         * Test all pairs at once
│         * Returns: DataFrame with columns [Pair, TestStat, PValue, Cointegrated, HedgeRatio]
│
│       Example Usage:
│       result = validation.engle_granger.engle_granger_test(prices['KO'], prices['PEP'])
│       if result.is_cointegrated():
│           print(f"Pairs are cointegrated, hedge ratio: {result.hedge_ratio}")
│
│   --- Only pairs passing this test (p_value < alpha) proceed to backtesting ---
│
├── trading/
│   │
│   ├── __init__.py
│   │
│   ├── spread.py
│   │   Responsibility: Calculate and manage the spread (trading unit)
│   │
│   │   Key Functions:
│   │   - calculate_spread(price1: pd.Series, price2: pd.Series, hedge_ratio: float) → pd.Series
│   │     * Formula: spread = price1 - (hedge_ratio * price2)
│   │     * Returns: Time series of spread values
│   │     * This is the actual "instrument" we trade
│   │
│   │   Example Usage:
│   │   spread = trading.spread.calculate_spread(prices['KO'], prices['PEP'], 0.87)
│   │
│   ├── signals.py
│   │   Responsibility: Generate trading signals based on spread statistics
│   │
│   │   Key Functions:
│   │   - calculate_zscore(spread: pd.Series, lookback: int = 20) → pd.Series
│   │     * Z-score = (spread - rolling_mean) / rolling_std
│   │     * Measures how many standard deviations spread is from mean
│   │     * Higher |z-score| = more extreme deviation = more trading opportunity
│   │
│   │   - generate_signals(zscore: pd.Series, long_threshold: float = -2, 
│   │                      short_threshold: float = 2) → pd.Series
│   │     * Logic:
│   │       - Signal = -1 (LONG the spread) when z-score < long_threshold
│   │       - Signal = 1 (SHORT the spread) when z-score > short_threshold
│   │       - Signal = 0 (FLAT/EXIT) when z-score crosses 0
│   │     * Returns: Series with {-1, 0, 1} values for each date
│   │
│   │   Example Usage:
│   │   zscore = trading.signals.calculate_zscore(spread)
│   │   signals = trading.signals.generate_signals(zscore)
│   │
│   ├── hedge_ratio.py
│   │   Responsibility: Calculate hedge ratios for position sizing
│   │
│   │   Key Functions:
│   │   - calculate_hedge_ratio(series1: pd.Series, series2: pd.Series) → float
│   │     * Simple regression: series1 ~ series2
│   │     * Returns beta coefficient (weight for series2)
│   │     * Used in spread calculation
│   │     * Already returned by engle_granger test, but can be recalculated
│   │
│   │   Example Usage:
│   │   hr = trading.hedge_ratio.calculate_hedge_ratio(prices['KO'], prices['PEP'])
│   │
│   └── backtest.py ⭐ CORE BACKTESTING ENGINE
│       Responsibility: Simulate trading strategy on historical data
│
│       Key Functions:
│       - run_backtest(pair: Tuple[str, str], prices: DataFrame, 
│                      start_date: str, end_date: str, initial_capital: float = 100000) → BacktestResult
│         * Input:
│           - pair: ('KO', 'PEP')
│           - prices: DataFrame with both tickers
│           - date range and initial capital
│         * Process:
│           1. Calculate spread using engle_granger hedge ratio
│           2. Generate z-score signals
│           3. Generate trading signals (-1, 0, 1)
│           4. Calculate position sizes (dollar-neutral)
│           5. Simulate daily P/L based on positions
│           6. Calculate equity curve
│           7. Extract individual trades
│         * Output: BacktestResult with:
│           - equity_curve: pd.Series of cumulative equity over time
│           - daily_returns: daily returns as %
│           - trades: DataFrame of individual trades
│             Columns: [entry_date, exit_date, entry_price1, entry_price2,
│                       exit_price1, exit_price2, position, pnl, return_pct]
│           - prices: prices DataFrame for the pair
│           - signals: signals for the pair
│           - spread: spread values
│           - zscore: z-score values
│
│       Example Usage:
│       result = trading.backtest.run_backtest(('KO', 'PEP'), prices, 
│                                               '2020-01-01', '2023-12-31')
│       print(result.equity_curve)
│       print(result.trades)
│
├── utils/
│   │
│   ├── __init__.py
│   │
│   ├── metrics.py
│   │   Responsibility: Calculate financial performance metrics
│   │
│   │   Key Functions:
│   │   - calculate_sharpe_ratio(returns: pd.Series, annual: bool = True) → float
│   │     * Formula: (mean_return / std_return) * sqrt(252) if annual else sqrt(252)
│   │     * Returns: Annualized Sharpe ratio (default)
│   │     * Higher is better (more return per unit of risk)
│   │
│   │   - calculate_max_drawdown(equity_curve: pd.Series) → float
│   │     * Peak-to-trough decline as percentage
│   │     * Formula: max((peak - value) / peak)
│   │     * Returns: Max DD as decimal (e.g., 0.15 = 15%)
│   │
│   │   - calculate_win_rate(trades: DataFrame) → float
│   │     * Percentage of trades that were profitable
│   │     * Returns: Value between 0 and 1
│   │
│   │   - calculate_profit_factor(trades: DataFrame) → float
│   │     * Sum of winning trades / Sum of losing trades
│   │     * > 1.0 means more wins than losses
│   │
│   │   - calculate_all_metrics(equity_curve, daily_returns, trades) → Dict
│   │     * Orchestrator: calls all metric functions
│   │     * Returns: Dict with all metrics for easy reporting
│   │
│   │   Example Usage:
│   │   metrics = utils.metrics.calculate_all_metrics(equity_curve, returns, trades)
│   │   print(f"Sharpe: {metrics['sharpe_ratio']}")
│   │   print(f"Max DD: {metrics['max_drawdown']}")
│   │
│   └── visualization.py
│       Responsibility: Create charts and visualizations for Streamlit
│       NOTE: This is a BACKEND utility, supporting frontend visualization
│
│       Key Functions:
│       - plot_equity_curve(equity_curve: pd.Series) → plotly.Figure
│         * Line chart of cumulative equity
│         * X-axis: dates, Y-axis: equity value
│         * Interactive Plotly figure ready for Streamlit
│
│       - plot_price_with_signals(prices: DataFrame, signals: pd.Series, 
│                                 pair: Tuple[str, str]) → plotly.Figure
│         * Two subplots: one for each leg of the pair
│         * Overlay entry/exit points as scatter markers
│         * Color code by position (long/short/flat)
│
│       - plot_zscore(zscore: pd.Series, signals: pd.Series) → plotly.Figure
│         * Z-score line chart
│         * Add horizontal lines at +2 and -2 (signal thresholds)
│         * Shade regions for buy/sell signals
│
│       - plot_equity_with_trades(equity_curve, trades) → plotly.Figure
│         * Equity curve with trade entry/exit markers
│
│       - plot_correlation_heatmap(prices: DataFrame) → plotly.Figure
│         * Heatmap of correlation matrix
│
│       Example Usage:
│       fig = utils.visualization.plot_equity_curve(equity_curve)
│       st.plotly_chart(fig)  # In Streamlit frontend
│

================================================================================
DATA FLOW DIAGRAM (The Pipeline)
================================================================================

User Input (tickers=['KO','PEP'], dates, params)
        ↓
[pipeline.py] → run_full_pipeline()
        ↓
[data/loader.py] → load_prices() → DataFrame(prices)
        ↓
[data/preprocessor.py] → prepare_features() → DataFrame(normalized_returns)
        ↓
[identification/kmeans.py] → fit_kmeans(n_clusters=3) → labels, centers
        ↓
[identification/dbscan.py] → fit_dbscan(eps=0.5) → labels (alternative)
        ↓
Extract Pairs from Clusters → List[(ticker1, ticker2), ...]
        ↓
[validation/engle_granger.py] → test_pairs() → DataFrame(test_results)
        ↓
Filter: Keep only cointegrated pairs (p_value < 0.05)
        ↓
For each valid pair:
    ├─ [validation/engle_granger.py] → hedge_ratio
    ├─ [trading/spread.py] → calculate_spread()
    ├─ [trading/signals.py] → calculate_zscore() + generate_signals()
    ├─ [trading/backtest.py] → run_backtest() → {equity, trades, metrics}
    └─ [utils/metrics.py] → calculate_all_metrics() → {sharpe, max_dd, ...}
        ↓
Aggregate Results → Dict with all results/metrics
        ↓
Return to pipeline.py (orchestrator)
        ↓
[frontend/] reads results and visualizes in Streamlit


================================================================================
IMPORT STRUCTURE (How modules reference each other)
================================================================================

Allowed Imports (following dependency hierarchy):

pipeline.py imports FROM:
  - from pairs_trading.data import loader, preprocessor
  - from pairs_trading.identification import kmeans, dbscan
  - from pairs_trading.validation import adf, engle_granger
  - from pairs_trading.trading import backtest, signals, spread, hedge_ratio
  - from pairs_trading.utils import metrics, visualization

trading/backtest.py imports FROM:
  - from pairs_trading.trading import signals, spread, hedge_ratio
  - from pairs_trading.validation import engle_granger
  - from pairs_trading.utils import metrics

frontend/pages/* imports FROM:
  - from pairs_trading import pipeline
  - from pairs_trading.data import loader
  - from pairs_trading.utils import visualization, metrics
  - (Any backend module as needed)

RULE: Lower-level modules (data, identification, validation) do NOT import 
from higher-level modules (trading, utils, pipeline). Data flows down, not up.


================================================================================
KEY INTEGRATION POINTS
================================================================================

1. FRONTEND → BACKEND Integration Points:

   Location: frontend/pages/*.py
   
   a) Data Exploration Page:
      prices = pairs_trading.data.loader.load_prices(tickers, start, end)
      stats = prices.describe()
      corr_matrix = prices.corr()
   
   b) Pair Identification Page:
      features = pairs_trading.data.preprocessor.prepare_features(prices)
      kmeans_result = pairs_trading.identification.kmeans.fit_kmeans(features, k=3)
      dbscan_result = pairs_trading.identification.dbscan.fit_dbscan(features, eps=0.5)
      # Extract pairs from cluster labels
   
   c) Cointegration Tests Page:
      eg_results = pairs_trading.validation.engle_granger.test_pairs(candidate_pairs, prices)
      valid_pairs = eg_results[eg_results['is_cointegrated'] == True]
   
   d) Backtest Results Page:
      pair = ('KO', 'PEP')
      backtest_result = pairs_trading.trading.backtest.run_backtest(pair, prices, ...)
      equity_fig = pairs_trading.utils.visualization.plot_equity_curve(backtest_result.equity_curve)
      st.plotly_chart(equity_fig)
   
   e) Performance Metrics Page:
      metrics = pairs_trading.utils.metrics.calculate_all_metrics(...)
      # Display in Streamlit metric cards

2. CACHING for Performance:
   
   Frontend should cache expensive operations:
   
   @st.cache_data
   def get_prices(ticker_list, start, end):
       return pairs_trading.data.loader.load_prices(ticker_list, start, end)
   
   @st.cache_data
   def get_clusters(prices, algorithm='kmeans', **kwargs):
       features = pairs_trading.data.preprocessor.prepare_features(prices)
       if algorithm == 'kmeans':
           return pairs_trading.identification.kmeans.fit_kmeans(features, **kwargs)
       else:
           return pairs_trading.identification.dbscan.fit_dbscan(features, **kwargs)


================================================================================
EXAMPLE: Complete Workflow
================================================================================

SCENARIO: User wants to find and backtest pairs from 3 tech stocks

1. Frontend calls:
   results = pairs_trading.pipeline.run_full_pipeline(
       tickers=['AAPL', 'MSFT', 'GOOG'],
       start_date='2020-01-01',
       end_date='2023-12-31',
       kmeans_k=2,
       dbscan_eps=0.5
   )

2. Inside pipeline.run_full_pipeline():
   
   a. Load prices:
      prices = data.loader.load_prices(['AAPL', 'MSFT', 'GOOG'], ...)
      # Returns: DataFrame(3 tickers x 1000 dates)
   
   b. Prepare features:
      features = data.preprocessor.prepare_features(prices)
      # Returns: DataFrame(3 tickers x 2) - PCA reduced, normalized
   
   c. Run clustering:
      kmeans_result = identification.kmeans.fit_kmeans(features, n_clusters=2)
      # Returns: labels=[0, 1, 0] meaning AAPL&GOOG in cluster 0, MSFT in cluster 1
      
      Pairs from cluster 0: [('AAPL', 'GOOG')]
   
   d. Test cointegration:
      candidate_pairs = [('AAPL', 'GOOG')]
      eg_results = validation.engle_granger.test_pairs(candidate_pairs, prices)
      # Returns: DataFrame with p_value=0.032 (COINTEGRATED!)
      
      valid_pairs = [('AAPL', 'GOOG')]
   
   e. Backtest each valid pair:
      for pair in valid_pairs:
          result = trading.backtest.run_backtest(pair, prices, ...)
          # Returns: BacktestResult with equity_curve, trades, etc.
          
          metrics = utils.metrics.calculate_all_metrics(
              result.equity_curve,
              result.daily_returns,
              result.trades
          )
          # Returns: {sharpe: 1.23, max_dd: 0.15, ...}
   
   f. Return everything:
      return {
          'prices': prices,
          'clustering': {'kmeans': kmeans_result, 'dbscan': dbscan_result},
          'valid_pairs': valid_pairs,
          'backtest_results': {('AAPL', 'GOOG'): result},
          'metrics': {('AAPL', 'GOOG'): metrics}
      }

3. Frontend visualizes:
   st.line_chart(results['backtest_results'][('AAPL', 'GOOG')].equity_curve)
   st.dataframe(results['backtest_results'][('AAPL', 'GOOG')].trades)
   st.metric("Sharpe Ratio", results['metrics'][('AAPL', 'GOOG')]['sharpe_ratio'])


================================================================================
DEVELOPMENT GUIDELINES
================================================================================

1. MODULARITY:
   - Each file should do ONE thing well
   - Functions should be pure when possible (same input = same output)
   - Avoid side effects (writing to files, printing to console in core logic)

2. ERROR HANDLING:
   - Validate inputs (ticker names, date ranges, numeric parameters)
   - Raise clear exceptions with helpful messages
   - Let frontend handle user-facing error messages

3. TESTING:
   - Unit tests for each function in isolation
   - Integration tests for pipeline.run_full_pipeline()
   - Test with known data (e.g., KO-PEP example)

4. DOCUMENTATION:
   - Docstrings for all functions
   - Type hints for all parameters and returns
   - Example usage in docstring

5. PERFORMANCE:
   - Avoid loading all data multiple times
   - Cache expensive computations in frontend
   - Consider vectorization for large datasets

6. VERSION CONTROL:
   - Commit backend code separately from frontend
   - Descriptive commit messages
   - Keep data/ folder in .gitignore (except sample data)


================================================================================
ADDING NEW FEATURES
================================================================================

Example: Add support for more clustering algorithms

1. Create new file: pairs_trading/identification/hierarchical.py

2. Implement function:
   def fit_hierarchical(features: DataFrame, n_clusters: int) -> HierarchicalResult:
       # Implementation
       return HierarchicalResult(labels=..., linkage_matrix=...)

3. Import in pipeline.py:
   from pairs_trading.identification import hierarchical

4. Add to pipeline.py logic:
   hierarchical_result = identification.hierarchical.fit_hierarchical(features, n_clusters)

5. Update frontend:
   in pair_identification.py, add option to sidebar radio button

That's it! Modular design makes adding features straightforward.


================================================================================
TROUBLESHOOTING
================================================================================

Issue: ModuleNotFoundError when running frontend
Fix: Ensure pairs-trading/ folder is in PYTHONPATH or use absolute imports

Issue: Data doesn't align across tickers
Fix: Check that loader.load_prices() handles mismatched date ranges

Issue: Clustering produces only 1 cluster
Fix: Check feature scaling in preprocessor.py, adjust n_clusters parameter

Issue: Backtest equity curve has NaN values
Fix: Check for missing prices data, ensure no division by zero in spread calc

Issue: P-values don't make sense
Fix: Verify test implementation matches statistical theory, compare with statsmodels

================================================================================
FINAL NOTES
================================================================================

This structure enables:
✓ Clear separation of concerns (data → identify → validate → trade → analyze)
✓ Easy testing of individual components
✓ Simple integration with Streamlit frontend
✓ Scalability for adding new features
✓ Reusable code across different projects
✓ Clear data flow and dependencies

Questions or unclear sections? Refer to INSTRUCTIONS.md in frontend/ folder
for additional context on the pipeline stages.

================================================================================
