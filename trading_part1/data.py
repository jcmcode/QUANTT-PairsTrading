""" 
        get data from yfinance for 2 stocks 
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
"""

import yfinance as yf
import pandas as pd
import os

def get_data(ticker1, ticker2, start_date, end_date):
  """
          - Create a function `get_data(ticker1, ticker2, start_date, end_date)` that:
          - Uses `yfinance` to download **Adjusted Close** prices for both tickers.  
          - Cleans missing data and aligns both time series by date.  
          - Returns a clean `DataFrame` with two columns (e.g., `KO`, `PEP`).
  """
  # Download **Adjusted Close** prices for both tickers
  df1 = yf.download(ticker1, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
  df2 = yf.download(ticker2, start=start_date, end=end_date, auto_adjust=False)['Adj Close']
  
  # Combine two dataframes into one dataframe
  # axis=1 (Stack by Columns): when dataframes have the same index, add columns of one dataframe to the right of other
  data = pd.concat([df1, df2], axis=1)
  data.columns = [ticker1, ticker2]
  
  # Clean missing data
  data.dropna(inplace=True) 
  
  return data
  
def save_data(ticker1, ticker2, data):
  """
      Save the data to `data/KO_PEP.csv` for reproducibility.
  """
  # Make a folder named data, continue if it already exists
  os.makedirs("data", exist_ok=True)
  file_path = f"data/{ticker1}_{ticker2}.csv"
  data.to_csv(file_path)
  
def summarize_data(ticker1, ticker2, data):
  """
      Add basic summary outputs:
      - Start/end dates  
      - Number of data points  
      - Correlation between the two assets
  """
  
  # data.index return list of dates
  # strftime: string format time, convert time into string
  start_date = data.index.min().strftime("%y-%m-%d")
  end_date = data.index.max().strftime("%y-%m-%d")
  num_points = len(data)
  corr = data[ticker1].corr(data[ticker2])
  
  print(f"Start date: {start_date}")
  print(f"End date: {end_date}")
  print(f"Number of data points: {num_points}")
  print(f"Correleation between {ticker1} and {ticker2}: {corr}")
  
  
if __name__ == "__main__":
    # Example usage — pull KO & PEP data
    ticker1 = "KO"
    ticker2 = "PEP"
    start_date = "2018-01-01"
    end_date = "2024-12-31"

    # get data for the two tickers 
    data = get_data(ticker1, ticker2, start_date, end_date)
    
    # save dataframe to .csv
    save_data(ticker1, ticker2, data)
    
    # print summarized information
    summarize_data(ticker1, ticker2, data)
    
    