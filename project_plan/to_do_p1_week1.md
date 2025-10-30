# Project Overview Part 1:

# What We Need To Build:

Data from yfinance -> hedge ratio calculation via linear regression -> calculating spread, moving average over different time periods (30d, 60d,1 y etc..) -> calculate z-score and signals for trading indications -> backtesting trading strategy with programmed trading logic using signals (z-score etc...), track different metrics -> visualize trading performance, and other important metrics

This is the general pipeline of what we need to build for the first stage of the project

## Tasks:

Get data from yfinance, put into df and output as csv and put csv into data folder

Use data from yfinance to calcualte hedge ratio of the 2 stocks

Start building functions for calculating the spread, calculate z-score

Work on generating trading signals based on the z-score

Start on backtesting trading logic 

Work on functions to track PnL, maximum drawdown, sharpe ratio, profit factor etc

Visualize performance


##  Team Task Breakdown — Week 1 (Part 1: Pairs Trading)

###  Goal for the Week
Develop the **core components** of the pairs trading pipeline:
- Pull price data for KO & PEP
- Calculate hedge ratio and spread
- Compute moving averages and z-scores
- Begin generating trading signals
- Lay the foundation for backtesting and visualization

---

### ** Member 1 — Data **
**Objective:** Build the data collection and preprocessing module.

**Tasks:**
- [ ] Create a function `get_data(ticker1, ticker2, start_date, end_date)` that:
  - Uses `yfinance` to download **Adjusted Close** prices for both tickers.  
  - Cleans missing data and aligns both time series by date.  
  - Returns a clean `DataFrame` with two columns (e.g., `KO`, `PEP`).
- [ ] Save the data to `data/KO_PEP.csv` for reproducibility.  
- [ ] Add basic summary outputs:
  - Start/end dates  
  - Number of data points  
  - Correlation between the two assets

**Deliverables:**
- File: `data.py`  
- Clean dataset saved to `/data/KO_PEP.csv`  


---

### ** Member 2 — Stats **
**Objective:** Quantify the long-term relationship between the two stocks.

**Tasks:**
- [ ] Import data from /data/KO_PEP.csv. (Wont be done right away so build what you can)
- [ ] Use `statsmodels.api.OLS` to calculate the **hedge ratio** (β) between KO and PEP.  
- [ ] Print the regression summary and store the hedge ratio in a variable.  
- [ ] Build a function `calculate_spread(data, ticker1, ticker2, hedge_ratio)` to compute the spread.  
- [ ] Compute **rolling means and standard deviations** of the spread using multiple lookback windows:
  - 30 days  
  - 60 days  
  - 252 days (≈1 year)
- [ ] Plot the spread with its rolling mean(s) to visualize mean reversion behavior. (Not necessary right away but if time this would be great) 

**Deliverables:**
- File: `hedge_ratio.py`  
- Calculated hedge ratio printed in console  
- Plot of spread + rolling averages (saved to `/data/plots/`) (If you plotted it not necessary right away just extra) 

---

### ** Member 3 — Signals **
**Objective:** Implement z-score calculation, signal generation, and backtesting skeleton.

**Tasks:**
- [ ] Create a function `calculate_zscore(spread, window)` that standardizes the spread:
  - z = (spread − rolling_mean) / rolling_std  
- [ ] Write signal logic:
  - **Long Entry:** `zscore < -2`  
  - **Short Entry:** `zscore > 2`  
  - **Exit:** `abs(zscore) < 0.5`  

**Deliverables:**
- File: `signals.py`  

---

### ** Member 4 — Backtest **
**Objective:** Begin building the backtesting framework to simulate trades and evaluate strategy performance.  

**Tasks:**
- [ ] Create a `backtest()` function that:
  - Takes in:
    - The price data (`/data/KO_PEP.csv`)
    - The trading signals from signals.py
    - Ticker names (`ticker1`, `ticker2`)
  - Simulates positions based on signals:
    - **Long spread:** Buy KO, Sell PEP when `zscore < -2`
    - **Short spread:** Sell KO, Buy PEP when `zscore > 2`
    - **Exit:** Close all positions when `|zscore| < 0.5`
  - Tracks daily returns, cumulative PnL, and equity curve over time.  
  - Returns `pnl` (daily profit/loss) and `equity` (cumulative performance).

- [ ] Add a `performance_metrics()` function that calculates:
  - **Total Return** = (Final equity − 1) × 100  
  - **Sharpe Ratio** = mean(daily returns) ÷ std(daily returns) × √252  
  - **Maximum Drawdown** = largest drop from a peak in the equity curve  
  - (Optional) **Profit Factor** = sum(profits) ÷ |sum(losses)|  

- [ ] Print key metrics in the console at the end of the run for now.

**Deliverables:**
- File: `backtest.py`  
- A functioning `backtest()` and `performance_metrics()` implementation  
- Console output showing:
  - Number of trades executed  
  - Total Return  
  - Sharpe Ratio  
  - Max Drawdown 

---

###  **End-of-Week Target**
By **Monday**, the team should be able to:
- ✅ Load and clean KO & PEP price data  
- ✅ Calculate and display hedge ratio  
- ✅ Compute spread & z-score  
- ✅ Generate preliminary trading signals  
- ⚙️ Have some backtesting logic built 

---


