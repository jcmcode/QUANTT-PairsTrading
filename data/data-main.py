#User requests a period of time and certain tickers from yfinance. 
# try and get most timly data points

# 1 minute data, 7 days - FANG, NVIDIA, TSMC, AVGO, QCOM
# 15 minute data, 60 days
# 60 minute 
# day by day, 
# look into autorun of the data saving

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib as plt
from datetime import datetime, timedelta
import os

#pull user requested stock tickers into a single list
def getDate(Interval):
    
    if(Interval == '1m'):
        print("Data is Limited to the last 7 days")
        end = datetime.now()
        start =  (datetime.now() - timedelta(days =7))
    elif(Interval == '15m' or Interval == '1h'):
        print("Data is Limited to the last 60 days")
        end = datetime.now().strftime("%Y-%m-%d")
        start =  (datetime.now() - timedelta(days =59))
    else:
        start = input("Enter start date (YYYY-MM-DD): ").strip()
        end   = input("Enter end date (YYYY-MM-DD): ").strip()

    return start, end

def data():
    
    raw = input("Enter all tickers seperated by commas: ")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]
    dataInterval = input("Data interval('1m', '15m', '1h', '1d'):")

    getStart, getEnd = getDate(dataInterval)

    print("\n")
    print("    ********************************")
    print(f"Data is from {getStart} to {getEnd}")
    print("    ********************************")
    print("\n")

    data = yf.download(tickers, start=getStart, end=getEnd, interval=dataInterval,auto_adjust=False)['Adj Close']
    data.index = pd.to_datetime(data.index)

    data = data.fillna(0)

    print(data.tail())
    return data
    
#dailyData()
data()
#saveData()