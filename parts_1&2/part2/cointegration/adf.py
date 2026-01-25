import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.stattools as stat
from scipy.stats import linregress

directory1 = 'C:/VSCode/QUANT-PairsTrading/QUANTT-PairsTrading/trading_part1/data/KO_PEP.csv'
df  = pd.read_csv(directory1)

def visual(): 
    
    df['Date'] = pd.to_datetime(df['Date'])

    plt.figure(figsize = (10,5))
    plt.plot(df['Date'], df['KO'] , label = 'KO')
    plt.plot(df['Date'], df['PEP'] , label = 'PEP')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Closing Price')
    plt.title('KO and PEP Adjsuted Closing Prices')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.scatter(df['KO'], df['PEP'], s = 3)
    plt.xlabel('KO')
    plt.ylabel('PEP')
    plt.show()


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

visual()
adf()
