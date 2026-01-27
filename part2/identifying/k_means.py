import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import requests
from io import StringIO

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

K_PRICE      = 10   
K_VOLATILITY = 10   
K_MOMENTUM   = 10   


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        response = requests.get(url, headers=headers)
        df = pd.read_html(StringIO(response.text))[0]
        tickers = df["Symbol"].tolist()
        return [t.replace(".", "-") for t in tickers]
    except Exception as e:
        print(f"Error fetching tickers: {e}")
        return ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA", "TSLA", "JPM", "XOM", "GLD", "TLT"]


def get_data(tickers):
    print(f"Total {len(tickers)} stocks")
    data = yf.download(tickers, start="2023-01-01", group_by='ticker', auto_adjust=True, progress=True, threads=True)
    
    price_series_list = []
    for t in tickers:
        try:
            if t in data:
                series = data[t]["Close"]
                series.name = t
                price_series_list.append(series)
        except: pass
    
    if price_series_list:
        prices = pd.concat(price_series_list, axis=1)
        prices = prices.dropna(axis=1, thresh=int(0.9 * len(prices)))
        prices = prices.ffill().dropna()
        return prices
    
    return pd.DataFrame()


def prepare_features(prices):
   
   #Price
    feat_price = prices.copy()

    #Volatility
    returns = prices.pct_change()
    feat_vol = returns.rolling(window=20).std().dropna()

    #Momentum
    feat_mom = prices.pct_change(periods=20).dropna()
    common_index = feat_vol.index.intersection(feat_mom.index)
    
    return {
        "Price": feat_price.loc[common_index],
        "Volatility": feat_vol.loc[common_index],
        "Momentum": feat_mom.loc[common_index]
    }


def cluster_and_plot(data_df, metric_name, k):
    print(f"\n {metric_name} (k={k}) ")
    
    # Normalize
    scaler = StandardScaler()
    X = data_df.T
    X_scaled = scaler.fit_transform(X.T).T
    
    # KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    
    # Plotting
    fig, axes = plt.subplots(k, 1, figsize=(10, 3*k), sharex=True)
    if k == 1: axes = [axes]
    
    dates = data_df.index
    
    for i in range(k):
        ax = axes[i]
        indices = np.where(labels == i)[0]
        cluster_data = X_scaled[indices]
        
    
        for series in cluster_data[:30]:
            ax.plot(dates, series, color='gray', alpha=0.1)
            
        center = km.cluster_centers_[i]
        ax.plot(dates, center, color='red', linewidth=2)
        
        ax.set_title(f"Cluster {i} ({len(indices)} stocks)")
        ax.set_ylabel(metric_name)
        ax.grid(True)
        
    plt.suptitle(f"Clustering by {metric_name} Patterns", fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return pd.Series(labels, index=data_df.columns, name=f"{metric_name}_Cluster")


if __name__ == "__main__":

    tickers = get_sp500_tickers()
    
    
    
    prices = get_data(tickers)
    
    if not prices.empty:
        features = prepare_features(prices)
        
        results_list = []
        
    
        res_price = cluster_and_plot(features["Price"], "Price", k=K_PRICE)
        results_list.append(res_price)
        
        res_vol = cluster_and_plot(features["Volatility"], "Volatility", k=K_VOLATILITY)
        results_list.append(res_vol)
        
        res_mom = cluster_and_plot(features["Momentum"], "Momentum", k=K_MOMENTUM)
        results_list.append(res_mom)
        
    
        final_df = pd.concat(results_list, axis=1)
        final_df.index.name = "Ticker"
        
        
        print(final_df.head(20))
        
        
        final_df.to_csv("clustering_results.csv")
        

    else:
        print("Error")
