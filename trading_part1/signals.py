#creates entry and exist signals based on z-score
import pandas as pd
import hedge_ratio as hr

def generate_signals(data: pd.DataFrame, ticker1: str, ticker2: str, days: int = 30) -> pd.DataFrame:
    """
    Calculate z-score and generate trading signals
    -1 = Spread high -> Short KO, Long PEP (Sell the spread)
    +1 = Spread low -> Long Ko, Short PEP (Buy the spread)
     0 = Exit position
    """
    
    # Calculate Spread
    hedge_ratio =  hr.hedge(data)
    St = hr.calculate_spread(data, ticker1, ticker2, hedge_ratio)

    # Calculate z-score
    rolling_mean_St = St.rolling(window=days).mean()
    rolling_std_St = St.rolling(window=days).std()
    Zt = (St - rolling_mean_St) / rolling_std_St

    # Generate signals
    signal = pd.Series(0, index = St.index)
    signal[Zt > 2] = -1 # Spread high (Sell spread)
    signal[Zt < -2] = 1 # Spread low (Buy spread)
    signal[Zt.abs() < 0.5] = 0  # Exit position

    # Return them
    return pd.DataFrame({
        'spread': St,
        'zscore': Zt,        
        'signal': signal
    })
