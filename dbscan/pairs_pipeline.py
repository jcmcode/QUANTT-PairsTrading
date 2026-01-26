import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import statsmodels.tsa.stattools as ts
import argparse
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------------
# CONFIGURATION & DEFAULTS
# --------------------------------------------------------------------------------

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD", "INTC", "QCOM",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "AXP", "V", "MA",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH",
    "PG", "KO", "PEP", "COST", "WMT", "TGT",
    "DIS", "NFLX", "CMCSA",
    "BA", "LMT", "RTX", "GE",
    "CAT", "DE",
    "MMM", "HON",
    "IBM", "CSCO"
]

# --------------------------------------------------------------------------------
# DATA ACQUISITION
# --------------------------------------------------------------------------------

def fetch_price_data(tickers, interval="1d", period="2y"):
    """
    Fetches historical price data using yfinance.
    """
    print(f"[INFO] Fetching data for {len(tickers)} tickers. Interval: {interval}, Period: {period}")
    
    try:
        data = yf.download(tickers, period=period, interval=interval, progress=False, group_by='ticker', auto_adjust=True)
    except Exception as e:
        print(f"[ERROR] Failed to download data: {e}")
        return pd.DataFrame()

    # Reshape to get a single Close price DataFrame
    # yfinance multi-ticker download with group_by='ticker' returns MultiIndex columns (Ticker, OHLCV)
    # We want a DataFrame where columns are Tickers and values are Close prices.
    
    price_df = pd.DataFrame()
    
    # Check if we got a MultiIndex (multiple tickers) or Single Index
    if len(tickers) == 1:
        # If single ticker provided (rare for pairs trading but possible)
        # yfinance normally just returns columns Open, High, Low, Close...
        price_df[tickers[0]] = data['Close']
    else:
        # Extract 'Close' for each ticker
        # data.columns is likely (Ticker, 'Close'), (Ticker, 'Open')...
        # We can iterate or use xs if the level names are set.
        # A safer generic way for recent yfinance versions:
        
        # If data columns are MultiIndex:
        if isinstance(data.columns, pd.MultiIndex):
            # Attempt to slice cross-section for 'Close'
            try:
                # Keep in mind yfinance structure can vary slightly by version.
                # Usually level 0 is Ticker, level 1 is Price Type, OR vice versa depending on 'group_by'
                # With group_by='ticker': Level 0 = Ticker, Level 1 = Price Type
                # Let's iterate manually to be safe or use proper slicing.
                
                # Faster approach: just iterate tickers
                for t in tickers:
                    if t in data:
                        # Depending on yf version, might be data[t]['Close']
                        col = data[t]['Close'] if 'Close' in data[t] else None
                        if col is not None:
                            price_df[t] = col
            except Exception as e:
                print(f"[WARN] Error parsing MultiIndex data: {e}. Trying alternative parsing.")
                # Fallback: maybe group_by didn't work as expected or 'auto_adjust' changed things
                pass
        else:
            # Maybe flat columns if something weird happened or 'Close' is already there?
            # If standard yfinance download without group_by='ticker', columns might be MultiIndex (Price Type, Ticker)
            # We used group_by='ticker', so Level 0 should be Ticker.
            pass

    # Basic cleanup
    if price_df.empty and not data.empty:
        # Retry assuming data might be (Price, Ticker) if group_by failed or was ignored
        try:
             # Try yfinance standard "Adj Close" or "Close"
             price_df = data['Close'] if 'Close' in data else data['Adj Close']
        except:
            pass

    # Drop fully empty columns
    price_df.dropna(axis=1, how='all', inplace=True)
    
    print(f"[INFO] Downloaded shape: {price_df.shape}")
    return price_df

def fetch_sp500_tickers():
    """
    Scrapes the list of S&P 500 tickers from Wikipedia.
    """
    print("[INFO] Fetching S&P 500 tickers from Wikipedia...")
    try:
        import requests
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        
        tables = pd.read_html(response.text)
        df_sp500 = tables[0]
        tickers = df_sp500['Symbol'].tolist()
        # Some tickers on Wiki use '.' instead of '-' (e.g. BRK.B -> BRK-B for yfinance)
        tickers = [t.replace('.', '-') for t in tickers]
        print(f"[INFO] Found {len(tickers)} S&P 500 tickers.")
        return tickers
    except Exception as e:
        print(f"[ERROR] Failed to fetch S&P 500 list: {e}")
        return []

def fetch_fundamentals(tickers):
    """
    Fetches Sector and Market Cap for a list of tickers for clustering.
    Returns a DataFrame indexed by Ticker.
    """
    print(f"[INFO] Fetching fundamentals for {len(tickers)} tickers (this may take a moment)...")
    data = []
    
    # Batching might be needed if list is huge, but yfinance Ticker object is 1-by-1 usually for info.
    # We can use yf.Tickers(string_list) but accessing .info is still often sequential or slow.
    # Optimization: Use 'fast_info' if available (newer yfinance).
    
    # To avoid API bans or super long waits, we'll try to be efficient.
    # yfinance 0.2+ has .fast_info which is faster.
    
    for t in tickers:
        try:
            ticker_obj = yf.Ticker(t)
            # Try fast_info first (no web scrape usually, implies standard reliable data)
            mcap = None
            if hasattr(ticker_obj, 'fast_info'):
                 mcap = ticker_obj.fast_info.get('market_cap')
            
            # If fast_info fails or is None, try .info (slower)
            if mcap is None:
                mcap = ticker_obj.info.get('marketCap')
                
            # Sector usually needs .info
            sector = ticker_obj.info.get('sector', 'Unknown')
            
            data.append({
                "Ticker": t,
                "MarketCap": mcap if mcap is not None else 0,
                "Sector": sector
            })
        except Exception:
            # excessive errors might mean rate limit.
            # Just push what we have.
            data.append({"Ticker": t, "MarketCap": 0, "Sector": "Unknown"})
            
    df = pd.DataFrame(data).set_index("Ticker")
    
    # Encode Sector? No, do that in build_features. Just return raw data here.
    return df

# --------------------------------------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------------------------------------

def compute_returns(price_df, method="log"):
    """
    Computes returns from price data.
    """
    if method == "log":
        return np.log(price_df / price_df.shift(1))
    else:
        return price_df.pct_change()

def build_features(returns_df, fundamentals_df=None, n_components=50):
    """
    Builds feature matrix using PCA on returns + Optional Fundamentals.
    """
    # 1. Cleaning
    clean_returns = returns_df.dropna(axis=0, how='any')
    
    if clean_returns.shape[0] < 50:
        print("[WARN] Returns history too short after dropping NaNs. Relaxing drop rule (fillna 0).")
        clean_returns = returns_df.fillna(0)
    
    valid_tickers = clean_returns.columns.tolist()
    
    # Filter fundamentals to match valid tickers
    if fundamentals_df is not None:
        # Reindex to ensure alignment
        fundamentals_df = fundamentals_df.reindex(valid_tickers).fillna({"MarketCap": 0, "Sector": "Unknown"})
    
    if len(valid_tickers) < 2:
        raise ValueError("Not enough tickers with data to proceed.")

    # 2. PCA
    n_comp_actual = min(n_components, clean_returns.shape[0], clean_returns.shape[1])
    
    pca = PCA(n_components=n_comp_actual)
    pca.fit(clean_returns) 
    
    # (n_stocks, n_components)
    pca_features = pca.components_.T 
    
    print(f"[INFO] Explained variance ratio sum: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # 3. Incorporate Fundamentals
    if fundamentals_df is not None:
        print("[INFO] Adding fundamental features (Market Cap + Sector)...")
        # Market Cap Log Scale
        mcap = np.log1p(fundamentals_df['MarketCap'].values).reshape(-1, 1)
        
        # Sector One-Hot
        # Using pandas get_dummies
        sector_dummies = pd.get_dummies(fundamentals_df['Sector'], prefix="Sec").astype(float)
        # Ensure alignment
        sector_features = sector_dummies.values
        
        # We need to balance weights?
        # PCA features are usually small values (loadings).
        # Standard scaling later will normalize everything, so raw magnitude doesn't matter too much 
        # BEFORE scaling, but we want scaling to treat them equally.
        
        # Combine: PCA (N, k) + Mcap (N, 1) + Sector (N, S)
        features = np.hstack([pca_features, mcap, sector_features])
    else:
        features = pca_features

    print(f"[INFO] Feature matrix (Stocks, Features): {features.shape}")
    
    # 4. Standardization
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    return X, valid_tickers

# --------------------------------------------------------------------------------
# CLUSTERING
# --------------------------------------------------------------------------------

def run_dbscan(X, eps=1.9, min_samples=3):
    """
    Runs DBSCAN clustering.
    Attempts to auto-tune eps if default fails to find clusters.
    """
    print(f"[INFO] Initial DBSCAN with eps={eps}, min_samples={min_samples}")
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(X)
    
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    # Simple Auto-Tune / Retry Logic
    if n_clusters == 0:
        print("[WARN] No clusters found with default settings. Attempting auto-tune...")
        
        # Estimate good scale from data
        # If we didn't compute it outside, do it here
        neigh = NearestNeighbors(n_neighbors=2).fit(X)
        dists, _ = neigh.kneighbors(X)
        avg_dist = np.mean(dists[:, 1])
        print(f"[DEBUG] Auto-tune base avg distance: {avg_dist:.2f}")
        
        # Search range around the avg distance
        # DBSCAN eps usually needs to be somewhat larger than avg NN dist to form clusters
        start_eps = max(0.5, avg_dist * 0.5)
        end_eps = avg_dist * 3.0
        step = 0.5
        
        eps_candidates = np.arange(start_eps, end_eps, step)
        min_samples_retry = 2
        
        best_labels = labels
        best_n_clusters = 0
        best_eps = eps
        
        for e in eps_candidates:
            # We also try min_samples=2 which is crucial for pairs
            # print(f"[INFO] Trying eps={e:.1f}...", end="\r")
            db_try = DBSCAN(eps=e, min_samples=min_samples_retry, metric='euclidean')
            l_try = db_try.fit_predict(X)
            nc_try = len(set(l_try)) - (1 if -1 in l_try else 0)
            
            # Heuristic: We want clusters, but not just 1 giant cluster containing everyone.
            # If > 50% of data is in one cluster, it's maybe too loose? 
            # For now, just maximize cluster count.
            
            if nc_try > best_n_clusters:
                best_n_clusters = nc_try
                best_labels = l_try
                best_eps = e
        
        labels = best_labels
        n_clusters = best_n_clusters
        n_noise = list(labels).count(-1)
        print(f"\n[INFO] Auto-tune selected eps={best_eps:.2f}, min_samples={min_samples_retry} with {n_clusters} clusters.")

    print(f"[INFO] Final DBSCAN: {n_clusters} clusters and {n_noise} noise points.")
    return labels

# --------------------------------------------------------------------------------
# VISUALIZATION
# --------------------------------------------------------------------------------

def plot_tsne(X, labels, tickers, output_file="tsne_clusters.png"):
    """
    Plots t-SNE visualization of the stocks.
    """
    print("[INFO] Computing t-SNE embeddings...")
    # Adjust perplexity if N is small
    n_samples = X.shape[0]
    perp = min(25, n_samples - 1) if n_samples > 1 else 1
    
    tsne = TSNE(n_components=2, perplexity=perp, learning_rate=200, random_state=1337, init='pca')
    X_embedded = tsne.fit_transform(X)
    
    plt.figure(figsize=(12, 10))
    
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Noise
            col = 'k'
            alpha = 0.1
            label_name = "Noise"
            marker = '.'
        else:
            alpha = 0.8
            label_name = f"Cluster {k}"
            marker = 'o'
        
        class_member_mask = (labels == k)
        xy = X_embedded[class_member_mask]
        
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], label=label_name if k != -1 else None, 
                    alpha=alpha, s=50 if k != -1 else 20, marker=marker)
        
        # Annotate some points if not noise
        if k != -1:
            # Annotate all members of clusters
            members = np.array(tickers)[class_member_mask]
            for i, txt in enumerate(members):
                plt.annotate(txt, (xy[i, 0], xy[i, 1]), fontsize=8, alpha=0.7)
    
    plt.title("t-SNE Visualization of Stock Clusters (PCA + DBSCAN)")
    plt.xlabel("t-SNE Dim 1")
    plt.ylabel("t-SNE Dim 2")
    
    # Legend only if not too many clusters
    if len(unique_labels) < 20:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[INFO] Saved t-SNE plot to {output_file}")
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[INFO] Saved t-SNE plot to {output_file}")
    plt.close()

def plot_cluster_sizes(labels, output_file="cluster_sizes.png"):
    """
    Plots a bar chart of cluster sizes.
    """
    import collections
    counts = collections.Counter(labels)
    # Filter noise if needed, but usually good to show noise count
    
    # Sort by cluster ID
    sorted_keys = sorted(counts.keys())
    sorted_counts = [counts[k] for k in sorted_keys]
    sorted_labels = [str(k) if k != -1 else "Noise" for k in sorted_keys]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(sorted_labels, sorted_counts, color=['red' if l=="Noise" else 'skyblue' for l in sorted_labels])
    
    plt.title('Cluster Member Counts')
    plt.xlabel('Cluster Number') # The user asked for 'Stocks in Cluster' on X? 
    # USER REQUEST: "plt.xlabel('Stocks in Cluster') plt.ylabel('Cluster Number');"
    # Wait, usually bar plots are: X=Cluster ID, Y=Count.
    # User asked:
    # "plt.title('Cluster Member Counts') plt.xlabel('Stocks in Cluster') plt.ylabel('Cluster Number');"
    # This implies a HORIZONTAL bar plot? Or maybe they mixed up labels. 
    # "Stocks in Cluster" sounds like "Count". "Cluster Number" sounds like the Category.
    # If Y is Cluster Number, it's horizontal.
    # Let's check intent. "Stocks in Cluster" is a Quantity. "Cluster Number" is an ID.
    # A standard vertical bar chart has Cluster ID on X and Count on Y.
    # A horizontal one has Cluster ID on Y and Count on X.
    # If user wants xlabel='Stocks in Cluster', that implies X axis is Quantity. -> Horizontal Plot.
    
    plt.close() # clear previous config
    
    plt.figure(figsize=(10, round(len(sorted_keys)*0.5 + 4)))
    y_pos = np.arange(len(sorted_keys))
    
    plt.barh(y_pos, sorted_counts, color=['red' if k == -1 else 'skyblue' for k in sorted_keys])
    plt.yticks(y_pos, sorted_labels)
    
    plt.title('Cluster Member Counts')
    plt.xlabel('Stocks in Cluster')
    plt.ylabel('Cluster Number')
    
    # Add counts at end of bars
    for i, v in enumerate(sorted_counts):
        plt.text(v + 0.5, i, str(v), color='black', va='center')
        
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"[INFO] Saved Cluster Size plot to {output_file}")
    plt.close()

# --------------------------------------------------------------------------------
# PAIR VALIDATION
# --------------------------------------------------------------------------------

def test_cointegration(series_a, series_b):
    """
    Runs Engle-Granger Coinregration test on log prices.
    Returns p-value.
    Null hypothesis: No cointegration.
    Low p-value (<0.05) -> Cointegrated.
    """
    # Use statsmodels coint
    # c_t: Constant and trend. 'c': constant only.
    score, pvalue, _ = ts.coint(series_a, series_b, autolag='AIC')
    return pvalue

def find_pairs_by_cluster(price_df, labels, tickers, cluster_size_limit=200):
    """
    Iterates through clusters to find valid pairs.
    """
    candidates = []
    
    # Map label -> list of tickers
    cluster_map = {}
    for t, lab in zip(tickers, labels):
        if lab == -1: continue
        cluster_map.setdefault(lab, []).append(t)
        
    print(f"[INFO] Validating pairs in {len(cluster_map)} clusters...")
    
    # Pre-compute log prices for coint test
    log_prices = np.log(price_df)
    
    # Pre-compute simple returns for correlation
    returns = price_df.pct_change().dropna()
    
    for cluster_id, members in cluster_map.items():
        if len(members) < 2:
            continue
        if len(members) > cluster_size_limit:
            print(f"[WARN] Cluster {cluster_id} size {len(members)} exceeds limit {cluster_size_limit}. Skipping to avoid explosion.")
            continue
            
        # Generate pairs
        # itertools.combinations
        from itertools import combinations
        for t1, t2 in combinations(members, 2):
            try:
                # Get price series
                # Align data just in case
                p1 = log_prices[t1].dropna()
                p2 = log_prices[t2].dropna()
                
                # Find common index
                common_idx = p1.index.intersection(p2.index)
                if len(common_idx) < 50:
                    continue # Not enough overlap
                
                s1 = p1.loc[common_idx]
                s2 = p2.loc[common_idx]
                
                # Cointegration Test
                p_val = test_cointegration(s1, s2)
                
                # Correlation
                r1 = returns[t1].loc[common_idx[1:]] # returns has 1 less row
                r2 = returns[t2].loc[common_idx[1:]]
                corr = r1.corr(r2)
                
                if p_val < 0.05:
                    candidates.append({
                        "Cluster": cluster_id,
                        "Ticker1": t1,
                        "Ticker2": t2,
                        "P_Value": p_val,
                        "Correlation": corr
                    })
            except Exception as e:
                # Catch singular matrix errors etc
                continue
                
    results_df = pd.DataFrame(candidates)
    if not results_df.empty:
        results_df.sort_values(by="P_Value", inplace=True)
        
    return results_df

# --------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pairs Trading Pipeline")
    parser.add_argument("--interval", type=str, default="1d", help="Data interval (1m, 5m, 1h, 1d)")
    parser.add_argument("--period", type=str, default="2y", help="Data lookback period")
    parser.add_argument("--eps", type=float, default=1.9, help="DBSCAN epsilon")
    parser.add_argument("--min_samples", type=int, default=3, help="DBSCAN min_samples")
    parser.add_argument("--components", type=int, default=50, help="PCA components")
    parser.add_argument("--sp500", action="store_true", help="Use S&P 500 tickers")
    parser.add_argument("--use_fundamentals", action="store_true", help="Fetch and use Market Cap/Sector")
    
    args = parser.parse_args()
    
    # 1. Define Universe
    if args.sp500:
        tickers = fetch_sp500_tickers()
        if not tickers:
            print("[WARN] S&P fetch failed, falling back to default list.")
            tickers = DEFAULT_TICKERS
    else:
        # Default big tech + others
        tickers = DEFAULT_TICKERS
    
    # 2. Fetch Prices
    price_df = fetch_price_data(tickers, interval=args.interval, period=args.period)
    
    if price_df.empty:
        print("[ERROR] No data fetched. Exiting.")
        sys.exit(1)
        
    # 3. Optional Fundamentals
    fund_df = None
    if args.use_fundamentals:
        # Only fetch for valid columns in price_df to save time
        available_tickers = price_df.columns.tolist()
        fund_df = fetch_fundamentals(available_tickers)
        
    # 4. Returns
    returns_df = compute_returns(price_df, method="log")
    
    # 5. Features
    try:
        X, valid_tickers = build_features(returns_df, fundamentals_df=fund_df, n_components=args.components)
    except Exception as e:
        print(f"[ERROR] Feature building failed: {e}")
        sys.exit(1)
        
    # 6. Clustering
    # Debug: Check nearest neighbor distances
    nbrs = NearestNeighbors(n_neighbors=2).fit(X)
    distances, indices = nbrs.kneighbors(X)
    mean_min_dist = np.mean(distances[:, 1])
    print(f"[DEBUG] Mean distance to nearest neighbor: {mean_min_dist:.4f}")
    
    labels = run_dbscan(X, eps=args.eps, min_samples=args.min_samples)
    
    # 7. Visualization
    plot_tsne(X, labels, valid_tickers)
    plot_cluster_sizes(labels)
    
    # 8. Pair Validation
    # Use original price_df for cointegration tests
    pairs_df = find_pairs_by_cluster(price_df[valid_tickers], labels, valid_tickers)
    
    # 9. Outputs
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    # Cluster Sizes
    df_labels = pd.DataFrame({"Ticker": valid_tickers, "Cluster": labels})
    cluster_counts = df_labels.groupby("Cluster").count()
    print("\nCluster Sizes:")
    print(cluster_counts)
    
    # Top Pairs
    print(f"\nTotal Valid Pairs Found (p_val < 0.05): {len(pairs_df)}")
    if not pairs_df.empty:
        print("\nTop 10 Pairs by Cointegration P-Value:")
        print(pairs_df.head(10).to_string(index=False))
        
        pairs_df.to_csv("pairs_candidates.csv", index=False)
        print("\n[INFO] Full list saved to 'pairs_candidates.csv'")
    else:
        print("\n[INFO] No cointegrated pairs found with current parameters.")
        
    df_labels.to_csv("clusters.csv", index=False)
    print("[INFO] Cluster assignments saved to 'clusters.csv'")

if __name__ == "__main__":
    main()
