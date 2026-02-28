#!/usr/bin/env python3
"""
Generate README figures from pipeline pickle artifacts.

Usage:
    python TransientCorrelation/scripts/generate_figures.py

Outputs 5 PNGs to docs/figures/:
    sector_heatmap.png      Pair counts by sector
    test_pass_rates.png     5-test individual pass rates
    score_distribution.png  Score histogram + classification breakdown
    algorithm_radar.png     OPTICS vs KMeans vs DBSCAN radar chart
    cumulative_pnl.png      Cumulative P&L for a top-scoring pair
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCREENER_DATA = os.path.join(REPO_ROOT, "TransientCorrelation", "screener", "data", "combined")
RESEARCH_DATA = os.path.join(REPO_ROOT, "TransientCorrelation", "research", "data")
OUT_DIR = os.path.join(REPO_ROOT, "docs", "figures")

# Shared style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})


def _load(path):
    if not os.path.exists(path):
        print(f"  [SKIP] Missing artifact: {path}")
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 1. Sector Heatmap
# ---------------------------------------------------------------------------

def fig_sector_heatmap():
    registry = _load(os.path.join(SCREENER_DATA, "pair_registry.pkl"))
    if registry is None:
        return
    sectors = sorted(set(registry["sector_1"].unique()) | set(registry["sector_2"].unique()))
    mat = pd.DataFrame(0, index=sectors, columns=sectors)
    for _, r in registry.iterrows():
        s1, s2 = r["sector_1"], r["sector_2"]
        mat.loc[s1, s2] += 1
        if s1 != s2:
            mat.loc[s2, s1] += 1

    # Shorten labels for readability
    short = {
        "Technology": "Tech",
        "Healthcare": "Health",
        "Energy": "Energy",
        "Financial Services": "Financials",
        "Industrials": "Industrials",
    }
    mat.index = [short.get(s, s) for s in mat.index]
    mat.columns = [short.get(s, s) for s in mat.columns]

    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(mat, annot=True, fmt="d", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(f"Sector Pair Counts ({len(registry):,} pairs, noise-adj freq >= 8%)")
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "sector_heatmap.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 2. Test Pass Rates
# ---------------------------------------------------------------------------

def fig_test_pass_rates():
    results = _load(os.path.join(SCREENER_DATA, "analysis_results.pkl"))
    if results is None:
        return
    tests = [
        ("adf_passed", "ADF Stationarity"),
        ("hl_passed", "Half-Life (5-60d)"),
        ("hurst_passed", "Hurst (< 0.5)"),
        ("vr_passed", "Variance Ratio"),
        ("rc_passed", "Rolling Corr Stability"),
    ]
    rates, labels = [], []
    for col, label in tests:
        if col in results.columns:
            rates.append(results[col].mean())
            labels.append(label)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    colors = sns.color_palette("Blues_d", len(rates))
    bars = ax.barh(labels, rates, color=colors)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Pass Rate")
    ax.set_title("Individual Test Pass Rates (5-Test Framework)")
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{rate:.1%}", va="center", fontsize=10)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "test_pass_rates.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 3. Score Distribution
# ---------------------------------------------------------------------------

def fig_score_distribution():
    results = _load(os.path.join(SCREENER_DATA, "analysis_results.pkl"))
    if results is None:
        return

    score_colors = {0: "#d32f2f", 1: "#f57c00", 2: "#fbc02d",
                    3: "#7cb342", 4: "#388e3c", 5: "#1b5e20"}
    cls_colors = {"strong": "#388e3c", "moderate": "#7cb342",
                  "weak": "#fbc02d", "fail": "#d32f2f"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Score histogram
    counts = results["score"].value_counts().sort_index()
    bar_c = [score_colors.get(s, "#666") for s in counts.index]
    axes[0].bar(counts.index, counts.values, color=bar_c)
    axes[0].set_xlabel("Validation Score (out of 5)")
    axes[0].set_ylabel("Number of Pairs")
    axes[0].set_title("Score Distribution")
    axes[0].set_xticks(range(6))
    for x, y in zip(counts.index, counts.values):
        axes[0].text(x, y + 20, str(y), ha="center", fontsize=9)

    # Classification breakdown
    cls_order = ["strong", "moderate", "weak", "fail"]
    cls_counts = results["classification"].value_counts().reindex(cls_order, fill_value=0)
    axes[1].bar(cls_counts.index, cls_counts.values,
                color=[cls_colors[c] for c in cls_order])
    axes[1].set_ylabel("Number of Pairs")
    axes[1].set_title("Classification Breakdown")
    for x, y in zip(range(len(cls_counts)), cls_counts.values):
        axes[1].text(x, y + 20, str(y), ha="center", fontsize=9)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "score_distribution.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 4. Algorithm Radar Chart
# ---------------------------------------------------------------------------

def fig_algorithm_radar():
    algo_defs = [
        ("OPTICS", ""),
        ("KMeans", "kmeans_"),
        ("DBSCAN", "dbscan_"),
    ]
    algo_colors = {"OPTICS": "#2ca02c", "KMeans": "#1f77b4", "DBSCAN": "#d62728"}

    metrics = {}
    for label, prefix in algo_defs:
        ch = _load(os.path.join(RESEARCH_DATA, f"{prefix}cluster_history.pkl"))
        pc = _load(os.path.join(RESEARCH_DATA, f"{prefix}pair_classification.pkl"))
        dur = _load(os.path.join(RESEARCH_DATA, f"{prefix}df_durations.pkl"))
        if ch is None or pc is None or dur is None:
            continue

        n_ts = ch["Datetime"].nunique()
        avg_clusters = ch.groupby("Datetime")["Cluster_ID"].apply(
            lambda x: x[x != -1].nunique()
        ).mean()
        noise_pct = (ch["Cluster_ID"] == -1).mean()
        stable_frac = (pc["Category"] == "stable_candidate").mean()
        transient_frac = (pc["Category"] == "transient").mean()
        avg_duration = dur["Duration_Hours"].mean()

        metrics[label] = {
            "Avg Clusters/ts": avg_clusters,
            "Stable Pairs %": stable_frac,
            "Transient Pairs %": transient_frac,
            "Avg Duration (h)": avg_duration,
            "Low Noise %": 1 - noise_pct,  # invert so higher = better
        }

    if not metrics:
        print("  [SKIP] No algorithm data found for radar chart")
        return

    df = pd.DataFrame(metrics).T
    # Normalize to [0, 1]
    for col in df.columns:
        mn, mx = df[col].min(), df[col].max()
        df[col] = (df[col] - mn) / (mx - mn) if mx > mn else 1.0

    dims = list(df.columns)
    n = len(dims)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    for algo in df.index:
        vals = df.loc[algo].tolist() + [df.loc[algo].iloc[0]]
        ax.plot(angles, vals, "o-", linewidth=2, label=algo, color=algo_colors[algo])
        ax.fill(angles, vals, alpha=0.1, color=algo_colors[algo])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dims, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.set_title("Algorithm Comparison (normalized 0-1, higher = better)",
                 fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "algorithm_radar.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 5. Cumulative P&L for a Top Pair
# ---------------------------------------------------------------------------

def fig_cumulative_pnl():
    sys.path.insert(0, os.path.join(REPO_ROOT, "TransientCorrelation"))
    from validation.pair_validation import compute_hedge_ratio, zscore_signals, simulate_spread_pnl

    results = _load(os.path.join(SCREENER_DATA, "analysis_results.pkl"))
    prices = _load(os.path.join(SCREENER_DATA, "prices.pkl"))
    if results is None or prices is None:
        return

    # Pick the top-scoring pair with the most trades and positive P&L
    candidates = results[
        (results["score"] >= 4) &
        (results["n_trades"] >= 2) &
        (results["total_pnl"] > 0)
    ].sort_values("total_pnl", ascending=False)

    if candidates.empty:
        candidates = results[results["score"] >= 4].sort_values("total_pnl", ascending=False)
    if candidates.empty:
        print("  [SKIP] No suitable pair for cumulative P&L chart")
        return

    row = candidates.iloc[0]
    a, b = row["ticker_a"], row["ticker_b"]

    if a not in prices.columns or b not in prices.columns:
        print(f"  [SKIP] Prices missing for {a}-{b}")
        return

    p = prices[[a, b]].dropna()
    n = len(p)
    cal_end = int(n * 0.67)
    p_cal = p.iloc[:cal_end]
    p_oos = p.iloc[cal_end:]

    beta, intercept, _ = compute_hedge_ratio(p_cal[a], p_cal[b], method="ols")
    spread_oos = p_oos[a] - beta * p_oos[b]
    sigs = zscore_signals(spread_oos, lookback=20, entry_z=2.0, exit_z=0.5)
    result = simulate_spread_pnl(spread_oos, sigs)
    pnl = result["pnl_series"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(pnl.index, pnl.values, color="steelblue", linewidth=1.5)
    ax.axhline(0, color="black", linestyle="--", alpha=0.3)
    ax.fill_between(pnl.index, pnl.values, 0,
                    where=pnl.values >= 0, alpha=0.15, color="green")
    ax.fill_between(pnl.index, pnl.values, 0,
                    where=pnl.values < 0, alpha=0.15, color="red")
    ax.set_title(f"Cumulative P&L — {a}-{b} (OOS, score {int(row['score'])})")
    ax.set_ylabel("Cumulative P&L ($)")
    ax.set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "cumulative_pnl.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}  ({a}-{b}, {result['n_trades']} trades, "
          f"P&L={result['total_pnl']:.2f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating figures → {OUT_DIR}/\n")

    for name, fn in [
        ("sector_heatmap", fig_sector_heatmap),
        ("test_pass_rates", fig_test_pass_rates),
        ("score_distribution", fig_score_distribution),
        ("algorithm_radar", fig_algorithm_radar),
        ("cumulative_pnl", fig_cumulative_pnl),
    ]:
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:
            print(f"  [ERROR] {e}")
    print("\nDone.")


if __name__ == "__main__":
    main()
