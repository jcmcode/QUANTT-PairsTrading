import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def saveData():
    tickers = ['FANG', 'NVDA', 'TSM', 'AVGO', 'QCOM']
    csv_file = "MinuteData.csv"

    # Download 1-minute data (Adj Close only)
    df_new = yf.download(tickers, start=(datetime.now() - timedelta(days = 1)), end=datetime.now() , interval='1m', auto_adjust=False, progress=False)['Adj Close']

    # Ensure datetime index
    df_new.index = pd.to_datetime(df_new.index)
    df_new.index.name = "Datetime"

    # Fill missing values
    df_new = df_new.fillna(0)

    # If file exists → append
    if os.path.exists(csv_file):
        df_old = pd.read_csv(csv_file, index_col="Datetime", parse_dates=True)

        df = pd.concat([df_old, df_new])
        df = df[~df.index.duplicated(keep="last")]
    else:
        df = df_new

    print(df.head())

    # Save back to CSV
    df.to_csv(csv_file)

if __name__ == "__main__":
    saveData()