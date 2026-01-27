import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stat
from scipy.stats import linregress
from data.data_main import data as get_data

#directory1 = 'C:\VSCode\QUANT-PairsTrading\QUANTT-PairsTrading\parts_1&2\trading\trading_part1\data\KO_PEP.csv'
#df  = pd.read_csv(directory1)

def adf_new(df_new):
    
    result =  linregress(df_new[df_new.columns[0]], df_new[df_new.columns[1]])
    print("Linear Regression")
    print(f"Hedge Ratio: {result.slope}")
    print(f"Intercept: {result.intercept}")
    print("**************")

    #Compute spread
    df_new['Spread'] = df_new[df_new.columns[1]] - (result.intercept + result.slope * df_new[df_new.columns[0]])

    adfStat, pVal, usedlag, nobs, criticalVal, icbest = stat.adfuller(df_new['Spread'])
    print("ADF Test")
    print(f"ADF Statistic: {adfStat}")
    print(f"p-value: {pVal}")
    print(criticalVal)

    if(pVal > 0.05):
        print("Result: Non-Stationary Series")
    elif(adfStat < criticalVal[0]):
        print("Result: Stationary Series with 99% Accuracy")
    elif(adfStat < criticalVal[1]):
        print("Result: Stationary Series with 95% Accuracy")
    elif(adfStat < criticalVal[2]):
        print("Result: Stationary Series with 90% Accuracy")

    print(df_new.head())

def visual(data): 
    
    plt.figure(figsize = (10,5))
    plt.plot(data.index, data[data.columns[1]] , label = f"{data.columns[1]}")
    plt.plot(data.index, data[data.columns[0]] , label = f"{data.columns[0]}")
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price')
    plt.title(f"{data.columns[0]} and {data.columns[1]} Adjsuted Closing Prices")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.scatter(data[data.columns[1]], data[data.columns[0]], s = 3)
    plt.xlabel(data.columns[1])
    plt.ylabel(data.columns[0])
    plt.show()

"""
def adf():
    result =  linregress(df['KO'], df['PEP'])
    print("Linear Regression")
    print(f"Hedge Ratio: {result.slope}")
    print(f"Intercept: {result.intercept}")
    print("**************")

    #Compute spread
    df['Spread'] = df['PEP'] - (result.intercept + result.slope * df['KO'])

    adfStat, pVal, usedlag, nobs, criticalVal, icbest = stat.adfuller(df['Spread'])
    print("ADF Test")
    print(f"ADF Statistic: {adfStat}")
    print(f"p-value: {pVal}")
    print(criticalVal)

    if(pVal > 0.05):
        print("Result: Non-Stationary Series")
    elif(adfStat < criticalVal[0]):
        print("Result: Stationary Series with 99% Accuracy")
    elif(adfStat < criticalVal[1]):
        print("Result: Stationary Series with 95% Accuracy")
    elif(adfStat < criticalVal[2]):
        print("Result: Stationary Series with 90% Accuracy")
"""

if __name__ == "__main__":
    data = get_data()
    adf_new(data)
    visual(data)
    
