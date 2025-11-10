#simulates trades, computes returns, and metrics for performance evaluation
import signals as singal
import hedge_ratio as hedge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Backtest function
def backtest(ticker1, ticker2):

    pd.options.display.float_format = '{:.6f}'.format

    #Load csv and singals into dataframe
    df = pd.read_csv('C:/VSCode/QUANT-PairsTrading/QUANTT-PairsTrading/trading_part1/data/KO_PEP.csv')
    df_signal = singal.generate_signals(df, ticker1, ticker2)

    print(df_signal['signal'].value_counts())

    print(df.head())
    df['ticker1 - 1'] = df[ticker1].shift(periods = +1)
    df.loc[0, 'ticker1 - 1'] = 0

    df['ticker2 - 1'] = df[ticker2].shift(periods = +1)
    df.loc[0, 'ticker2 - 1'] = 0

    #Calculate % returns
    df['KO Returns'] = (df[ticker1] / df['ticker1 - 1']) - 1
    df['PEP Returns'] = (df[ticker2] / df['ticker2 - 1']) - 1

    #Create new dataframe foe return data 
    df_pairsReturn = df[['KO Returns', 'PEP Returns']].copy()
    df_pairsReturn['zscore singal'] = df_signal['signal'].values

    #Assign weightings to each retrun based off zscore(long or short position)
    df_pairsReturn['KO Weight'] = np.where(df_pairsReturn['zscore singal'] == - 1,-1, np.where(df_pairsReturn['zscore singal'] == 1, 1, 0 ))
    df_pairsReturn['PEP Weight'] = -1 * df_pairsReturn['KO Weight']

    #Calculate Daily returns and equity curve
    df_pairsReturn['Portfolio Daily Return'] = ( df_pairsReturn['KO Weight'] * df_pairsReturn['KO Returns']) + ( df_pairsReturn['PEP Weight'] * df_pairsReturn['PEP Returns'])
    df_pairsReturn['Equity Curve'] = (1 + df_pairsReturn['Portfolio Daily Return']).cumprod()
    
    #View Results
    print(df_pairsReturn.tail())
    print(df_pairsReturn['Portfolio Daily Return'].describe())
    print(df_pairsReturn['zscore singal'].value_counts())
    df_pairsReturn['Equity Curve'].plot()
    df_pairsReturn['Portfolio Daily Return'].plot()
    #df_pairsReturn['KO Returns'].plot()
    #df_pairsReturn['PEP Returns'].plot()
    plt.show()

    return df_pairsReturn['Portfolio Daily Return'], df_pairsReturn['Equity Curve']
    #Track position
    #Long Spread when zscore < -2, buy coke, sell pepsi, coke is undervalued, pepsi is expensive
    #Short Spread when zscore > 2, buy pepsi, sell coke 
    #Exit - close all positions sell long , buy short 

# performance metrics fucntion
def performance_metrics():
    
    #Call backtest 
    pnl, equity = backtest('KO', 'PEP')

    #Total retunn is the last element in data frame
    totalReturn = (equity.iloc[-1] - 1) * 100

    #Calculate sharpe ratio, check that standard deviation is non-zero
    if pnl.std() != 0:
        sharpe = pnl.mean()/pnl.std() * np.sqrt(252)
    else:
        sharpe = np.nan

    peak = equity.cummax()
    drawdown = (peak - equity)/peak
    max_drawdown = drawdown.max() * 100  # in %

    # Count changes in position (excluding 0 → 0)
    positions = pnl.copy()
    positions[pnl != 0] = 1
    positions[pnl == 0] = 0
    trades = positions.diff().abs().sum()

    # Print metrics
    print(f"Total Return (%): {totalReturn:.4f}%")
    print(f"Annualized Sharpe Ratio: {sharpe:.4f}")
    print(f"Maximum Drawdown (%): {max_drawdown:.4f}%")
    print(f"Total Number of Trades: {int(trades)}")

    print(equity.head(10))
    print(equity.tail(10))

    #Total return - use most recent equity value
    #Sharpe Ratio - mean(pnl) / std(pnl) x Sqrt(252)
    #Max drawdown - Largest change in peaks(Day to day)
    #Total number or trades executed
    #Print all to consol

performance_metrics()