#creates entry and exist signals based on z-score
import pandas as pd

def calc_zscore(St: pd.Series, days: int = 30) -> pd.Series:
    """
     Calculate z-score for given spread
    """
    rolling_mean_St = St.rolling(window=days).mean()
    rolling_std_St = St.rolling(window=days).std()
    Zt = (St - rolling_mean_St) / rolling_std_St
    return Zt


def generate_signals(Zt: pd.Series) -> pd.Series:
    """
    Generate trading signals based on z-score 
    -1 = Spread high -> Short KO, Long PEP (Sell the spread)
    +1 = Spread low -> Long Ko, Short PEP (buy the spread)
     0 = Exit position
    """
    signal = pd.Series(0, index = Zt.index)
    signal[Zt > 2] = -1 # Spread high
    signal[Zt < -2] = 1 # Spread low
    signal[Zt.abs() < 0.5] = 0  # Exit position
    return signal