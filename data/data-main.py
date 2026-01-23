#User requests a period of time and certain tickers from yfinance. 
# try and get most timly data points

# 1 minute data, 7 days - FANG, NVIDIA, TSMC, AVGO, QCOM
# 15 minute data, 60 days
# 60 minute 
# day by day, 
# look into autorun of the data saving

import yfinance as yf
import pandas as pd
import numbers as np
import matplotlib as plt

#pull user requested stock tickers into a single list
def getTickers():
    raw = input("Enter all tickers seperated by commas: ")
    tickers = [t.strip().upper() for t in raw.split(",") if t.strip()]

    start = input("Enter start date (YYYY-MM-DD): ").strip()
    end   = input("Enter end date (YYYY-MM-DD): ").strip()

    request = (tickers, start, end)

    #print(request)

    return request
#download one min data from each ticker 
def oneMinData():
    tickers, start, end = getTickers()

    data = yf.download(tickers, start='2025-01-01', end='2026-01-01', interval='1m' ,auto_adjust=False)['Adj Close']
    data.index = pd.to_datetime(data.index)

    data = data.fillna(0)

    print(data.head())
    
    return data

##download data from each ticker 
def fifteenMinData():
    tickers, start, end = getTickers()

    data = yf.download(tickers, start='2025-01-01', end='2026-01-01', interval='15m' ,auto_adjust=False)['Adj Close']
    data.index = pd.to_datetime(data.index)

    data = data.fillna(0)

    print(data.head())
    
    return data
        
#download data from each ticker 
def sixtyMinData():
    tickers, start, end = getTickers()

    data = yf.download(tickers, start='2025-01-01', end='2026-01-01', interval='1h' ,auto_adjust=False)['Adj Close']
    data.index = pd.to_datetime(data.index)

    data = data.fillna(0)

    print(data.head())

    return data

#download data from each ticker 
def dailyData():
    tickers, start, end = getTickers()

    data = yf.download(tickers, start='2025-01-01', end='2026-01-01',auto_adjust=False)['Adj Close']
    data.index = pd.to_datetime(data.index)

    data = data.fillna(0)

    print(data.head())
    return

def data():
    dataInterval = input("Data interval('1m', '15m', '1h', '1d'):")

    tickers, start, end = getTickers()

    data = yf.download(tickers, start='2025-01-01', end='2026-01-01', interval= dataInterval,auto_adjust=False)['Adj Close']
    data.index = pd.to_datetime(data.index)

    data = data.fillna(0)

    print(data.head())
    return
    
dailyData()