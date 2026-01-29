import sys
import os
sys.path.append(os.path.dirname(__file__))

import pandas as pd
import matplotlib.pyplot as plt
from engle_granger import engle_granger_test


df = pd.read_csv("trading_part1/data/KO_PEP.csv")

def visual(): 
    print(df.tail())
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

def run_eg():
    print("Engle–Granger Cointegration Test")
    print("**************")

    coint_t, pval, crit = engle_granger_test(df["KO"], df["PEP"])

    print(f"Test Statistic: {coint_t}")
    print(f"p-value: {pval}")
    print("Critical Values (1%, 5%, 10%):")
    print(crit)

    if pval > 0.05:
        print("No evidence of cointegration (fail to reject null)")
    elif coint_t < crit[0]:
        print("Cointegrated with 99% confidence")
    elif coint_t < crit[1]:
        print("Cointegrated with 95% confidence")
    elif coint_t < crit[2]:
        print("Cointegrated with 90% confidence")

visual()
run_eg()
