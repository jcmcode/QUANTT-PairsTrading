"""
QUANTT PAIRS TRADING STREAMLIT DASHBOARD - INSTRUCTIONS & CONTEXT
==================================================================

This file provides comprehensive context for developing and maintaining this Streamlit dashboard.
Use this as a reference for understanding the project scope, architecture, and integration points.

PROJECT OVERVIEW
================

Dashboard Purpose:
  Showcase a complete end-to-end pairs trading pipeline that demonstrates:
  1. Data ingestion & exploration
  2. Identifying potential pairs using clustering (K-Means, DBSCAN)
  3. Validating pairs with cointegration tests (ADF, Engle-Granger)
  4. Trading strategy execution through backtesting
  5. Performance analysis with financial metrics

User Journey:
  Home → Data → Pair Identification → Validation → Backtest → Metrics
  
This is meant to be a **portfolio piece** and **learning dashboard** that shows the complete
workflow from raw data to trading strategy results.

TECHNICAL STACK
===============

Framework: Streamlit (Python-native, no JavaScript needed)
Visualization: Plotly (interactive) + Matplotlib (static)
Navigation: streamlit-option-menu (sidebar menu)
Data: Pandas, NumPy, SciPy

Running the Dashboard:
  streamlit run frontend/main.py

EXISTING CODEBASE TO INTEGRATE
==============================

Part 1: Data
-----------
Location: part2/data_code/ticker_data.py
Purpose: Load historical price data
Expected Output: DataFrame with ticker symbols as columns, dates as index, adjusted close prices as values
Integration: Use in data_exploration.py to display price charts, stats, correlations

Part 2: Pair Identification (Clustering)
-----------------------------------------
K-Means:
  Location: part2/identifying/k_means.py
  Purpose: Cluster assets to find similar price behavior
  Inputs: Feature matrix (typically price returns or normalized prices), number of clusters (k)
  Expected Output: Cluster labels for each ticker, cluster centers
  Integration: Display 2D scatter plot (use PCA for dimensionality reduction), show clusters with colors
  
DBSCAN:
  Location: part2/identifying/dbscan.py
  Purpose: Density-based clustering to find natural groupings without pre-specifying cluster count
  Inputs: Feature matrix, eps (epsilon - neighborhood radius), min_samples
  Expected Output: Cluster labels (-1 for noise points), cluster centers
  Integration: Display 2D scatter plot with noise points highlighted differently

Pair Extraction Logic:
  - Each cluster contains multiple tickers
  - Extract all possible pairs (ticker combinations) from the same cluster
  - These become "candidate pairs" for further validation
  - Display as a table showing: Pair, Cluster ID, Distance between tickers, Correlation

Part 3: Pair Validation (Cointegration)
----------------------------------------
ADF Test (Augmented Dickey-Fuller):
  Location: part2/cointegration/adf.py
  Purpose: Test if a single time series is stationary
  Inputs: Time series (price or spread)
  Expected Output: Test statistic, p-value, critical values
  Integration: Run on each individual ticker, show results table with "Stationary?" column
  
Engle-Granger Test:
  Location: part2/cointegration/engle_granger.py
  Purpose: Test if two series are cointegrated (move together long-term)
  Inputs: Two price time series
  Expected Output: Test statistic, p-value, cointegrating relationship (hedge ratio)
  Integration: Run on all candidate pairs, show results table with p-values
  
Valid Pairs Filtering:
  - Filter pairs where Engle-Granger p-value < 0.05 (or user-configurable alpha)
  - These are the pairs that pass the cointegration test
  - Only these pairs proceed to backtesting (mean-reversion assumption holds)

Part 4: Backtesting
-------------------
Location: trading/trading_part1/backtest.py, signals.py, spread.py, hedge_ratio.py
Purpose: Simulate trading the identified pairs
Key Components:
  - Spread Calculation: price1 - (hedge_ratio * price2)
  - Z-Score Signals: (spread - mean) / std, generates signals when |z-score| > 2
  - Position Sizing: Dollar-neutral (long one leg, short the other by same dollar amount)
  - Entry/Exit: Enter when z-score crosses threshold, exit when it reverts to mean
  
Expected Output:
  - Equity curve: cumulative portfolio value over time
  - Trade log: entry date, exit date, entry price, exit price, P/L
  - Position history: which dates were in which positions
  - Daily returns: daily P/L as percentage
  
Integration Points in Dashboard:
  1. Backtest Results Page:
     - Show equity curve chart (line chart with date on x-axis)
     - Overlay entry/exit points on price chart (green triangles for entries, red for exits)
     - Display z-score line chart with +2/-2 threshold bands
     - Show individual trade details in table format
  
  2. Performance Metrics Page:
     - Calculate Sharpe Ratio: (avg daily return / std of daily return) * sqrt(252)
     - Calculate Max Drawdown: max peak-to-trough decline
     - Calculate Win Rate: (winning trades / total trades) * 100
     - Calculate P/L: (final equity - initial equity)
     - Additional metrics: Sortino ratio, Calmar ratio, profit factor, recovery period

DATA FLOW DIAGRAM
=================

Raw Price Data (CSV/yfinance)
        ↓
[Data Exploration Page] - Display prices, stats, correlations
        ↓
Feature Extraction (returns, normalized prices, etc.)
        ↓
[Pair Identification Page] - K-Means & DBSCAN clustering
        ↓
Candidate Pairs List
        ↓
[Cointegration Tests Page] - ADF & Engle-Granger validation
        ↓
Valid Pairs (cointegrated)
        ↓
[Backtest Results Page] - Simulate trading each pair
        ↓
Trade Results (equity curves, trades, metrics)
        ↓
[Performance Metrics Page] - Aggregate statistics and risk analysis

PAGE-BY-PAGE REQUIREMENTS
==========================

1. HOME PAGE
   Status: Skeleton complete, content done
   Purpose: Show pipeline overview and key concepts
   Features:
     - Pipeline diagram (5 boxes showing stages)
     - Key concepts section (pairs trading, cointegration explanation)
     - Dashboard features list
     - Quick start instructions

2. DATA EXPLORATION PAGE
   Status: Skeleton complete, implementation TODO
   Purpose: Display raw data and exploration statistics
   Features Needed:
     ✓ Metrics: Total tickers count, data period, number of trading days
     ✓ Available tickers list
     ✓ Price chart: Line chart showing multiple ticker prices over time (with toggle/multi-select)
     ✓ Correlation heatmap: Show correlation between all tickers
     ✓ Summary stats: Table with min, max, mean, std, returns for each ticker
   Data Sources:
     - Use load_tickers() from utils/data_loader.py
     - Use load_price_data() for price charts
     - Use compute_correlations() for heatmap
     - Use compute_summary_statistics() for stats table

3. PAIR IDENTIFICATION PAGE
   Status: Skeleton complete, implementation TODO
   Purpose: Visualize clustering results
   Features Needed:
     ✓ Algorithm selector: Radio buttons for K-Means, DBSCAN, or Compare Both
     ✓ K-Means section:
       - 2D scatter plot (first two PCA components) colored by cluster
       - Slider for k (number of clusters) - range 2-10
       - Cluster size distribution table
     ✓ DBSCAN section:
       - 2D scatter plot colored by cluster (noise points in different color)
       - Slider for eps (0.1 to 2.0)
       - Slider for min_samples (2 to 10)
       - Cluster size distribution table
     ✓ Candidate pairs table:
       - Columns: Pair (Ticker1-Ticker2), Cluster ID, Correlation, Distance
       - Sortable, filterable
   Data Sources:
     - Use load_clustering_results() from utils/data_loader.py
     - Implement run_kmeans() and run_dbscan() in data_loader for live computation

4. COINTEGRATION TESTS PAGE
   Status: Skeleton complete, implementation TODO
   Purpose: Validate statistical properties of pairs
   Features Needed:
     ✓ Test explanation section: Explain ADF and Engle-Granger tests
     ✓ Significance level slider (alpha): 0.01 to 0.10, default 0.05
     ✓ ADF test results table:
       - Columns: Ticker, Test Statistic, P-value, Stationary? (boolean)
       - Color code: green if stationary, red if not
     ✓ Engle-Granger test results table:
       - Columns: Pair, Test Statistic, P-value, Cointegrated? (boolean), Hedge Ratio
       - Color code: green if cointegrated, red if not
     ✓ Spread visualization for selected pair:
       - Left: Spread time series (line chart)
       - Right: Scatter plot of the two prices
       - Show mean and ±1/2 std bands on spread chart
     ✓ Valid pairs summary:
       - Table of pairs that passed cointegration test
       - These are the pairs that will be backtested
   Data Sources:
     - Use load_cointegration_results() from utils/data_loader.py
     - Implement run_adf_tests() and run_engle_granger_tests() for live computation

5. BACKTEST RESULTS PAGE
   Status: Skeleton complete, implementation TODO
   Purpose: Show trading strategy execution and results
   Features Needed:
     ✓ Pair selector: Dropdown of all pairs that passed cointegration test
     ✓ Date range selector (optional)
     ✓ Equity curve metrics: Total Return %, Final Value, Peak Value
     ✓ Main equity curve chart: Line chart showing cumulative value over time
       - Include peak line or highlighting
     ✓ Price charts with signals (two side-by-side):
       - Left: First leg prices
       - Right: Second leg prices
       - Overlay entry points (green triangles)
       - Overlay exit points (red triangles)
       - Show position direction (long/short) with colored background
     ✓ Z-score signal chart:
       - Line chart of z-score over time
       - Horizontal lines at +2 and -2
       - Shaded regions for entry signals (red for negative, green for positive)
     ✓ Trade-by-trade details table:
       - Columns: Entry Date, Exit Date, Signal Type (Long/Short), P/L $, Return %, Duration (days)
       - Sortable, filterable
       - Color code: green rows for profitable trades, red for losses
     ✓ Position history breakdown:
       - Long trades section: Count, average return, duration
       - Short trades section: Count, average return, duration
   Data Sources:
     - Use load_backtest_results(pair) from utils/data_loader.py
     - Implement run_backtest() for live computation if pair hasn't been backtested yet

6. PERFORMANCE METRICS PAGE
   Status: Skeleton complete, implementation TODO
   Purpose: Aggregate financial statistics across all pairs
   Features Needed:
     ✓ Key metrics cards (for selected pair or portfolio):
       - Total Return %
       - Sharpe Ratio (annualized)
       - Maximum Drawdown %
       - Win Rate %
     ✓ Multi-pair comparison table:
       - Columns: Pair, Total Return %, Sharpe Ratio, Max Drawdown %, Win Rate %, Num Trades, Profit Factor, Cumulative P/L $
       - Sortable by any column
       - Color-coded metrics (green for good, red for bad)
     ✓ Drawdown analysis:
       - Left: Drawdown line chart over time
       - Right: Return distribution histogram (daily or trade-level returns)
     ✓ Return analysis:
       - Left: Monthly returns heatmap (months on x, years on y, values colored)
       - Right: Cumulative returns chart (growth of $1000 initial investment)
     ✓ Trade statistics:
       - Cards: Total Trades, Winning Trades, Losing Trades, Breakeven Trades
       - Cards: Avg Winner %, Avg Loser %, Profit Factor, Avg Duration
     ✓ Risk-adjusted metrics:
       - Cards: Sharpe Ratio, Sortino Ratio, Calmar Ratio (with explanations)
     ✓ Summary statistics table:
       - Complete breakdown of all key metrics
   Calculation Notes:
     - Sharpe Ratio = (mean_daily_return / std_daily_return) * sqrt(252)
     - Max Drawdown = max((peak - value) / peak)
     - Win Rate = winning_trades / total_trades
     - Sortino Ratio = similar to Sharpe but only uses downside volatility
     - Profit Factor = sum(winning_trades) / sum(losing_trades)
   Data Sources:
     - Use load_performance_metrics() from utils/data_loader.py

INTEGRATION CHECKLIST
=====================

Before implementing each feature, ensure:
  ☐ Data loader function exists in utils/data_loader.py
  ☐ Helper visualization function exists in utils/helpers.py
  ☐ Page skeleton exists in pages/
  ☐ All TODOs in the page are documented
  
When implementing:
  ☐ Use st.cache_data or st.cache_resource for expensive computations
  ☐ Add error handling with try/except and st.error()
  ☐ Use st.spinner() for long-running operations
  ☐ Format numbers with proper decimals and units (%, $, etc.)
  ☐ Add help text with st.info() or st.help() for complex concepts
  ☐ Use plotly for interactive charts, matplotlib for static ones
  ☐ Follow the existing color scheme and styling

COMMON INTEGRATION PATTERNS
===========================

Pattern 1: Load external Python code
  # At the top of page file
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).parent.parent.parent))
  from part2.identifying.k_means import KMeans  # Example import

Pattern 2: Call existing function and display results
  result = existing_function(parameters)
  st.dataframe(result)  # If it's a DataFrame
  st.plotly_chart(visualization)  # If it's a Plotly figure

Pattern 3: Interactive parameter tuning
  n_clusters = st.slider("Number of Clusters", 2, 10, 3)
  # Re-run computation with new parameter
  result = recompute(n_clusters)
  st.plotly_chart(result)

Pattern 4: Caching expensive computations
  @st.cache_data
  def load_and_compute_clusters(ticker_list):
      # Expensive computation here
      return result
  
  result = load_and_compute_clusters(tuple(ticker_list))

STYLING NOTES
=============

Color Scheme:
  - Primary: #1f77b4 (blue)
  - Success: Green (use st.success() or color="green")
  - Warning/Error: Red
  - Neutral Background: #f0f2f6

Custom CSS is in main.py under st.markdown(..., unsafe_allow_html=True)
  - .main-header: Center-aligned blue titles
  - .pipeline-section: Light gray box with padding for overview sections

Typography:
  - Page titles: st.markdown("# Title")
  - Section headers: st.markdown("## Section")
  - Subsections: st.markdown("### Subsection")

Layout:
  - Use st.columns() for side-by-side layouts
  - Use st.tabs() if comparing multiple options
  - Use st.expander() for collapsible sections
  - Use st.metric() for key statistics

ASSUMPTIONS & NOTES
===================

1. Data Structure:
   - Prices are stored as CSV files or can be loaded via yfinance
   - DataFrame format: dates as index, tickers as columns, values are adjusted close prices
   - Returns are calculated as (price_today / price_yesterday) - 1

2. Clustering:
   - Features are pre-processed (normalized prices, returns, PCA-reduced, etc.)
   - Clusters are assumed to contain similar-behaving assets
   - Pairs can be any combination of assets within the same cluster

3. Cointegration:
   - Spreads are assumed to be stationary if they pass the tests
   - Hedge ratio from Engle-Granger test defines the spread: price1 - hedge_ratio * price2
   - Only cointegrated pairs are suitable for mean-reversion trading

4. Trading:
   - Z-score signals: long when z < -2, short when z > 2, exit when z crosses 0
   - Position sizing: dollar-neutral (equal dollar amounts long/short)
   - No transaction costs or slippage modeled (can be added later)
   - Backtests assume ideal execution (entry at signal price, exit at signal price)

5. Performance Metrics:
   - All returns are calculated as daily returns
   - Sharpe ratio assumes 252 trading days per year
   - Drawdown is calculated as peak-to-trough
   - P/L can be displayed in absolute ($) or relative (%) terms

FUTURE ENHANCEMENTS
====================

Phase 2 Ideas:
  - Parameter optimization (Grid search for z-score thresholds, hedge ratios, etc.)
  - Portfolio statistics (correlation between different pairs, portfolio-level Sharpe ratio)
  - Walk-forward backtesting (train on early period, test on later period)
  - Monte Carlo simulations of strategy
  - Risk management widgets (stop-loss levels, position sizing)
  - Deployment info (how to paper trade or live trade this strategy)
  - Sensitivity analysis (vary parameters and see impact on returns)

Phase 3:
  - Live data integration (update prices automatically)
  - Signal generation for live trading
  - Position tracking and management
  - Execution layer integration (connect to broker APIs)

DEBUGGING TIPS
==============

1. Check data loading:
   - Verify CSV files exist at expected paths
   - Print first few rows: st.dataframe(df.head())
   - Check for missing values: st.write(df.isnull().sum())

2. Check clustering:
   - Verify feature matrix has correct shape
   - Visualize clusters with scatter plot
   - Check if clusters are meaningful (not all in one cluster, not all separate)

3. Check cointegration:
   - Verify p-values make sense (between 0 and 1)
   - Manually check a few spreads are stationary
   - Compare test results with other statistical software

4. Check backtesting:
   - Verify equity curve is monotonically increasing or decreasing (no jumps)
   - Check that equity curve starts at ~1.0 (100% of initial capital)
   - Verify trades table has matching entry/exit dates (no orphaned trades)
   - Spot-check manual P/L calculation for a few trades

5. Performance issues:
   - Use @st.cache_data for expensive computations
   - Avoid loading all data when showing examples (use sample)
   - Consider pre-computing results and storing in files/database

CONTACTS & RESOURCES
====================

Project Repository: /Users/jack/Documents/code/QUANTT-PairsTrading
Main Implementation: part2/ and trading/
Dashboard: frontend/

Streamlit Docs: https://docs.streamlit.io/
Plotly Docs: https://plotly.com/python/
Pairs Trading Resources: See resources/ folder in project root
"""
