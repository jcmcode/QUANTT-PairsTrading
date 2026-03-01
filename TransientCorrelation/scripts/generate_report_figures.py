#!/usr/bin/env python3
"""
Generate additional report figures from pipeline pickle artifacts.

Outputs PNGs to docs/figures/:
    pipeline_diagram.png            Full methodology flowchart
    validation_lift.png             5x lift: clustered vs random pass rates
    backtest_comparison.png         Baseline / Enhanced / Kalman profitability
    noise_adjusted_scatter.png      Noise-adjusted vs naive frequency scatter
    pca_variance.png                PCA cumulative variance explained
    formation_duration_hist.png     Formation duration distribution
    walkforward_diagram.png         Walk-forward split schematic
    frequency_distribution.png      Co-cluster frequency distribution with 8% cutoff
    multi_pnl_grid.png              Multiple P&L curves for top pairs
    intrasector_crosssector_box.png Intra vs cross-sector score distributions
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCREENER_DATA = os.path.join(REPO_ROOT, "TransientCorrelation", "screener", "data", "combined")
RESEARCH_DATA = os.path.join(REPO_ROOT, "TransientCorrelation", "research", "data")
OUT_DIR = os.path.join(REPO_ROOT, "docs", "figures")

sys.path.insert(0, os.path.join(REPO_ROOT, "TransientCorrelation"))

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
# 1. Pipeline Diagram
# ---------------------------------------------------------------------------

def fig_pipeline_diagram():
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.3, 2.0, "Hourly\nPrices", "#e3f2fd"),
        (2.3, 2.0, "9 Features\nper Ticker", "#e8f5e9"),
        (4.3, 2.0, "StandardScaler\n→ PCA (90%)", "#fff3e0"),
        (6.3, 2.0, "OPTICS\nClustering", "#fce4ec"),
        (8.3, 2.0, "Co-Cluster\nFrequency", "#f3e5f5"),
        (10.3, 2.0, "Noise-Adj\nFilter (≥8%)", "#e0f7fa"),
        (12.3, 2.0, "3,643 Pair\nRegistry", "#fff9c4"),
    ]

    for x, y, text, color in boxes:
        box = FancyBboxPatch((x, y), 1.6, 1.0, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.8, y + 0.5, text, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    # Arrows between boxes
    for i in range(len(boxes) - 1):
        x1 = boxes[i][0] + 1.6
        x2 = boxes[i + 1][0]
        y = 2.5
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # Second row
    boxes2 = [
        (8.3, 0.3, "5-Test\nValidation", "#e8f5e9"),
        (10.3, 0.3, "2,148\nTradeable", "#c8e6c9"),
        (12.3, 0.3, "Z-Score\nBacktest", "#a5d6a7"),
    ]

    for x, y, text, color in boxes2:
        box = FancyBboxPatch((x, y), 1.6, 1.0, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor="#333", linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + 0.8, y + 0.5, text, ha="center", va="center",
                fontsize=8.5, fontweight="bold")

    # Arrow down from Registry to Validation
    ax.annotate("", xy=(9.1, 1.3), xytext=(13.1, 2.0),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    # Arrow right from Validation to Tradeable
    ax.annotate("", xy=(10.3, 0.8), xytext=(9.9, 0.8),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))
    # Arrow right from Tradeable to Backtest
    ax.annotate("", xy=(12.3, 0.8), xytext=(11.9, 0.8),
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.5))

    # Results annotation
    ax.text(13.1, -0.15, "64% Profitable\n(Kalman, 10bps)",
            ha="center", va="center", fontsize=9, fontweight="bold",
            color="#2e7d32",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f5e9", edgecolor="#2e7d32"))

    ax.set_title("Full Pipeline: From Hourly Prices to Trading Results", fontsize=14, pad=15)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "pipeline_diagram.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 2. Validation Lift (Clustered vs Random)
# ---------------------------------------------------------------------------

def fig_validation_lift():
    # Phase 1 transient validation results (from PROJECT_GUIDE / notebook outputs)
    # Clustered: 26/657 = 4.0%, Random: 7/889 = 0.8%
    categories = ["Clustered Pairs\n(n = 657)", "Random Pairs\n(n = 889)"]
    pass_rates = [4.0, 0.8]
    colors = ["#2e7d32", "#bdbdbd"]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    bars = ax.bar(categories, pass_rates, color=colors, width=0.45, edgecolor="#333", linewidth=1.2)

    for bar, rate in zip(bars, pass_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f"{rate:.1f}%", ha="center", va="bottom", fontsize=18, fontweight="bold")

    # Add lift annotation
    ax.annotate("5× lift", xy=(0.5, 3.0), fontsize=20, fontweight="bold",
                color="#d32f2f", ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#ffebee", edgecolor="#d32f2f"))

    ax.set_ylabel("Transient Validation Pass Rate (%)", fontsize=13)
    ax.set_title("Method Validation: Clustered vs Random Pairs\n(Phase 1, 40 Semiconductors)",
                 fontsize=15, fontweight="bold")
    ax.set_ylim(0, 5.5)
    ax.tick_params(axis='both', labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "validation_lift.png")
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 3. Backtest Strategy Comparison
# ---------------------------------------------------------------------------

def fig_backtest_comparison():
    er = _load(os.path.join(SCREENER_DATA, "enhanced_results.pkl"))
    if er is None:
        return

    # Compute profitability for pairs with any trades
    strategies = []

    # Baseline
    bl = er[er["baseline_n_trades"] > 0]
    bl_prof = (bl["baseline_pnl"] > 0).mean() * 100
    strategies.append(("Baseline\n(OLS, z=2.0, no costs)", bl_prof, len(bl)))

    # Enhanced
    enh = er[er["enhanced_n_trades"] > 0]
    enh_prof = (enh["enhanced_pnl"] > 0).mean() * 100
    strategies.append(("Enhanced\n(OLS, opt z, 10bps)", enh_prof, len(enh)))

    # Kalman
    kal = er[er["kalman_n_trades"] > 0]
    kal_prof = (kal["kalman_pnl"] > 0).mean() * 100
    strategies.append(("Kalman\n(Kalman β, opt z, 10bps)", kal_prof, len(kal)))

    labels = [s[0] for s in strategies]
    values = [s[1] for s in strategies]
    counts = [s[2] for s in strategies]
    colors = ["#90a4ae", "#42a5f5", "#2e7d32"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="#333", linewidth=1)

    for bar, val, n in zip(bars, values, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=15, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                f"n={n}", ha="center", va="center", fontsize=10, color="white", fontweight="bold")

    ax.set_ylabel("OOS Profitability Rate (%)")
    ax.set_title("Backtest Profitability: Three Strategy Comparison\n(Top 50 Pairs)")
    ax.set_ylim(0, 80)
    ax.axhline(50, color="#999", linestyle="--", alpha=0.5, label="50% (random)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper left", framealpha=0.8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "backtest_comparison.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 4. Noise-Adjusted vs Naive Frequency Scatter
# ---------------------------------------------------------------------------

def fig_noise_adjusted_scatter():
    registry = _load(os.path.join(SCREENER_DATA, "pair_registry.pkl"))
    if registry is None:
        return

    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(registry["raw_freq"], registry["noise_adj_freq"],
               alpha=0.25, s=12, color="#1565c0", edgecolors="none")

    # Diagonal line (y=x)
    lims = [0, max(registry["raw_freq"].max(), registry["noise_adj_freq"].max()) * 1.05]
    ax.plot(lims, lims, "--", color="#999", linewidth=1, label="y = x (no correction)")

    # 8% threshold line
    ax.axhline(0.08, color="#d32f2f", linestyle="-", linewidth=1, alpha=0.7, label="8% threshold")

    ax.set_xlabel("Naive Frequency (raw count / total timestamps)")
    ax.set_ylabel("Noise-Adjusted Frequency")
    ax.set_title("Noise-Adjusted vs Naive Co-Clustering Frequency\n(3,643 pairs above threshold)")
    ax.legend(loc="upper left")
    ax.set_xlim(0, lims[1])
    ax.set_ylim(0, lims[1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "noise_adjusted_scatter.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 5. PCA Variance Explained
# ---------------------------------------------------------------------------

def fig_pca_variance():
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    ts_df = _load(os.path.join(SCREENER_DATA, "ts_df.pkl"))
    if ts_df is None:
        return

    features_to_cluster = [
        'Returns', 'Vol_Short', 'Beta_SPX_Short', 'Beta_Sector_Short',
        'RSI', 'Momentum_5H', 'Vol_Regime_Shift',
        'Beta_SPX_Regime_Shift', 'Beta_Sector_Regime_Shift',
    ]

    # Take a single representative timestamp snapshot
    timestamps = ts_df.index.get_level_values("Datetime").unique()
    # Use middle timestamp for representative snapshot
    mid_ts = timestamps[len(timestamps) // 2]
    snapshot = ts_df.loc[mid_ts][features_to_cluster].dropna()

    if len(snapshot) < 5:
        print("  [SKIP] Not enough data for PCA variance plot")
        return

    scaled = StandardScaler().fit_transform(snapshot)
    pca = PCA().fit(scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    components = range(1, len(cumvar) + 1)

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Bar chart for individual variance
    ax.bar(components, pca.explained_variance_ratio_, color="#90caf9",
           edgecolor="#1565c0", linewidth=0.5, label="Individual", alpha=0.8)

    # Line for cumulative
    ax.plot(components, cumvar, "o-", color="#d32f2f", linewidth=2,
            markersize=6, label="Cumulative")

    # 90% threshold
    ax.axhline(0.90, color="#2e7d32", linestyle="--", linewidth=1.5,
               alpha=0.7, label="90% threshold")

    # Find where cumvar crosses 90%
    n_components_90 = np.argmax(cumvar >= 0.90) + 1
    ax.axvline(n_components_90, color="#2e7d32", linestyle=":", alpha=0.5)
    ax.text(n_components_90 + 0.15, 0.5,
            f"{n_components_90} components\nretain ≥90%",
            fontsize=9, color="#2e7d32", fontweight="bold")

    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Variance Explained")
    ax.set_title("PCA Variance Explained (9 Clustering Features)")
    ax.set_xticks(list(components))
    ax.legend(loc="center right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "pca_variance.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}  ({n_components_90} components for 90% variance)")


# ---------------------------------------------------------------------------
# 6. Formation Duration Histogram
# ---------------------------------------------------------------------------

def fig_formation_duration():
    durations = _load(os.path.join(SCREENER_DATA, "df_durations.pkl"))
    if durations is None:
        return

    dur_hours = durations["Duration_Hours"].astype(float)
    dur_hours = dur_hours[dur_hours > 0]

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Use log-spaced bins
    bins = np.logspace(np.log10(0.5), np.log10(dur_hours.max()), 50)
    ax.hist(dur_hours, bins=bins, color="#42a5f5", edgecolor="#1565c0",
            linewidth=0.3, alpha=0.85)
    ax.set_xscale("log")

    # Add median and mean lines
    median_val = dur_hours.median()
    mean_val = dur_hours.mean()
    ax.axvline(median_val, color="#d32f2f", linestyle="-", linewidth=2,
               label=f"Median: {median_val:.0f}h")
    ax.axvline(mean_val, color="#2e7d32", linestyle="--", linewidth=2,
               label=f"Mean: {mean_val:.1f}h")

    ax.set_xlabel("Formation Duration (hours, log scale)")
    ax.set_ylabel("Count")
    ax.set_title(f"Cluster Formation Duration Distribution\n({len(dur_hours):,} formations, Phase 2)")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "formation_duration_hist.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}  (median={median_val:.0f}h, mean={mean_val:.1f}h)")


# ---------------------------------------------------------------------------
# 7. Walk-Forward Split Diagram
# ---------------------------------------------------------------------------

def fig_walkforward_diagram():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 5.5)
    ax.axis("off")

    n_splits = 5
    total_width = 9.0
    window_frac = 0.80
    cal_frac = 0.67
    window_width = total_width * window_frac
    cal_width = window_width * cal_frac
    oos_width = window_width * (1 - cal_frac)
    step = (total_width - window_width) / (n_splits - 1)

    for i in range(n_splits):
        y = n_splits - 1 - i
        x_start = 0.5 + i * step

        # Calibration block
        rect_cal = plt.Rectangle((x_start, y), cal_width, 0.6,
                                  facecolor="#42a5f5", edgecolor="#1565c0",
                                  linewidth=1.5, alpha=0.85)
        ax.add_patch(rect_cal)
        ax.text(x_start + cal_width / 2, y + 0.3, f"Cal {i+1}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

        # OOS block
        rect_oos = plt.Rectangle((x_start + cal_width, y), oos_width, 0.6,
                                  facecolor="#ef5350", edgecolor="#c62828",
                                  linewidth=1.5, alpha=0.85)
        ax.add_patch(rect_oos)
        ax.text(x_start + cal_width + oos_width / 2, y + 0.3, f"OOS {i+1}",
                ha="center", va="center", fontsize=9, color="white", fontweight="bold")

    # Legend
    cal_patch = mpatches.Patch(facecolor="#42a5f5", edgecolor="#1565c0",
                               label="Calibration (67%)")
    oos_patch = mpatches.Patch(facecolor="#ef5350", edgecolor="#c62828",
                               label="Out-of-Sample (33%)")
    ax.legend(handles=[cal_patch, oos_patch], loc="lower right", fontsize=10,
              framealpha=0.9)

    # Time arrow
    ax.annotate("", xy=(9.8, -0.3), xytext=(0.3, -0.3),
                arrowprops=dict(arrowstyle="->", color="#333", lw=1.5))
    ax.text(5.0, -0.45, "Time →", ha="center", fontsize=10, color="#333")

    ax.set_title("Walk-Forward Validation: 5 Rolling Splits\n(Window = 80% of data, sliding forward)",
                 fontsize=13, fontweight="bold", pad=10)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "walkforward_diagram.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 8. Frequency Distribution
# ---------------------------------------------------------------------------

def fig_frequency_distribution():
    # Load ALL pairs (not just registry) to show the full distribution
    pair_co_cluster_freq = _load(os.path.join(SCREENER_DATA, "pair_co_cluster_freq.pkl"))
    cluster_history = _load(os.path.join(SCREENER_DATA, "cluster_history.pkl"))
    total_windows = _load(os.path.join(SCREENER_DATA, "total_windows.pkl"))

    if pair_co_cluster_freq is None or cluster_history is None:
        return

    # Compute noise-adjusted frequency for ALL pairs
    # First get non-noise timestamps per ticker
    non_noise = cluster_history[cluster_history["Cluster_ID"] != -1]
    ticker_non_noise_ts = non_noise.groupby("Ticker")["Datetime"].apply(set).to_dict()

    freqs = []
    for (a, b), count in pair_co_cluster_freq.items():
        ts_a = ticker_non_noise_ts.get(a, set())
        ts_b = ticker_non_noise_ts.get(b, set())
        valid = len(ts_a & ts_b)
        if valid > 0:
            freqs.append(count / valid)

    freqs = np.array(freqs)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    bins = np.linspace(0, min(freqs.max(), 0.6), 60)
    ax.hist(freqs, bins=bins, color="#42a5f5", edgecolor="#1565c0",
            linewidth=0.3, alpha=0.85)

    # 8% threshold
    ax.axvline(0.08, color="#d32f2f", linestyle="-", linewidth=2,
               label=f"8% threshold\n({(freqs >= 0.08).sum():,} pairs pass)")

    n_below = (freqs < 0.08).sum()
    ax.text(0.04, ax.get_ylim()[1] * 0.92,
            f"{n_below:,}\nfiltered out", ha="center", fontsize=9, color="#666",
            fontweight="bold")
    ax.text(0.16, ax.get_ylim()[1] * 0.92,
            f"{(freqs >= 0.08).sum():,}\nretained", ha="center", fontsize=9,
            color="#d32f2f", fontweight="bold")

    ax.set_xlabel("Noise-Adjusted Co-Clustering Frequency")
    ax.set_ylabel("Number of Pairs")
    ax.set_title(f"Co-Clustering Frequency Distribution\n({len(freqs):,} total pairs)")
    ax.legend(loc="upper right", fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "frequency_distribution.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}  (total={len(freqs)}, pass 8%={(freqs >= 0.08).sum()})")


# ---------------------------------------------------------------------------
# 9. Multiple P&L Grid
# ---------------------------------------------------------------------------

def fig_multi_pnl_grid():
    from validation.pair_validation import compute_hedge_ratio, zscore_signals, simulate_spread_pnl

    er = _load(os.path.join(SCREENER_DATA, "enhanced_results.pkl"))
    prices = _load(os.path.join(SCREENER_DATA, "prices.pkl"))
    ar = _load(os.path.join(SCREENER_DATA, "analysis_results.pkl"))
    if er is None or prices is None or ar is None:
        return

    # Pick 6 diverse pairs: mix of profitable/sectors, sorted by kalman_pnl
    candidates = er[er["kalman_n_trades"] > 0].copy()
    candidates = candidates.sort_values("kalman_pnl", ascending=False)

    # Try to get diverse sectors
    selected = []
    seen_sectors = set()
    for _, row in candidates.iterrows():
        sector_key = f"{row['sector_1']}-{row['sector_2']}"
        if len(selected) < 6:
            selected.append(row)
            seen_sectors.add(sector_key)
    # If we don't have 6, just take top 6
    if len(selected) < 6:
        selected = list(candidates.head(6).itertuples(index=False))

    n_pairs = min(6, len(selected))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()

    for idx in range(n_pairs):
        row = selected[idx]
        a, b = row["ticker_a"], row["ticker_b"]
        ax = axes[idx]

        if a not in prices.columns or b not in prices.columns:
            ax.text(0.5, 0.5, f"{a}-{b}\n(no price data)", transform=ax.transAxes,
                    ha="center", va="center")
            continue

        p = prices[[a, b]].dropna()
        n = len(p)
        cal_end = int(n * 0.67)
        p_cal = p.iloc[:cal_end]
        p_oos = p.iloc[cal_end:]

        try:
            # Use Kalman if available, else OLS
            if pd.notna(row.get("kalman_beta", None)) and row["kalman_beta"] != 0:
                beta = row["kalman_beta"]
            else:
                beta, _, _ = compute_hedge_ratio(p_cal[a], p_cal[b], method="ols")

            spread_oos = p_oos[a] - beta * p_oos[b]

            # Use optimized z-params if available
            entry_z = row.get("opt_entry_z", 2.0)
            exit_z = row.get("opt_exit_z", 0.5)
            lookback = int(row.get("opt_lookback", 20))

            sigs = zscore_signals(spread_oos, lookback=lookback,
                                  entry_z=entry_z, exit_z=exit_z)
            result = simulate_spread_pnl(spread_oos, sigs, cost_per_trade=0.001)
            pnl = result["pnl_series"]

            ax.plot(pnl.index, pnl.values, color="steelblue", linewidth=1.2)
            ax.axhline(0, color="black", linestyle="--", alpha=0.3)
            ax.fill_between(pnl.index, pnl.values, 0,
                            where=pnl.values >= 0, alpha=0.15, color="green")
            ax.fill_between(pnl.index, pnl.values, 0,
                            where=pnl.values < 0, alpha=0.15, color="red")

            # Look up score from analysis_results
            pair_key = f"{a}-{b}" if f"{a}-{b}" in ar["pair"].values else f"{b}-{a}"
            score_row = ar[ar["pair"] == pair_key]
            score = int(score_row["score"].iloc[0]) if not score_row.empty else "?"

            pnl_val = result["total_pnl"]
            n_trades = result["n_trades"]
            sector_info = f"{row['pair_type']}"
            ax.set_title(f"{a}–{b} (score {score}, {sector_info})\n"
                         f"P&L: ${pnl_val:.1f}, {n_trades} trades",
                         fontsize=9, fontweight="bold")
        except Exception as e:
            ax.text(0.5, 0.5, f"{a}-{b}\nError: {str(e)[:30]}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=8)

        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=8)

    # Hide unused axes
    for idx in range(n_pairs, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Out-of-Sample Cumulative P&L — Top Pairs (Kalman Strategy, 10bps costs)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, "multi_pnl_grid.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# 10. Intra-sector vs Cross-sector Score Box Plot
# ---------------------------------------------------------------------------

def fig_intrasector_crosssector():
    ar = _load(os.path.join(SCREENER_DATA, "analysis_results.pkl"))
    if ar is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Box plot of scores
    intra = ar[ar["pair_type"] == "intra-sector"]["score"]
    cross = ar[ar["pair_type"] == "cross-sector"]["score"]

    bp = axes[0].boxplot([intra, cross], labels=["Intra-Sector", "Cross-Sector"],
                          patch_artist=True, widths=0.4)
    bp["boxes"][0].set_facecolor("#42a5f5")
    bp["boxes"][1].set_facecolor("#ef5350")
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    axes[0].set_ylabel("Validation Score (0-5)")
    axes[0].set_title("Score Distribution by Pair Type")

    # Add means
    axes[0].text(1, intra.mean() + 0.15, f"μ={intra.mean():.2f}", ha="center",
                 fontsize=9, color="#1565c0", fontweight="bold")
    axes[0].text(2, cross.mean() + 0.15, f"μ={cross.mean():.2f}", ha="center",
                 fontsize=9, color="#c62828", fontweight="bold")

    # Stacked bar chart: classification breakdown
    for pair_type, color, offset in [("intra-sector", "#42a5f5", -0.2),
                                      ("cross-sector", "#ef5350", 0.2)]:
        subset = ar[ar["pair_type"] == pair_type]
        total = len(subset)
        cls_order = ["strong", "moderate", "weak", "fail"]
        counts = subset["classification"].value_counts().reindex(cls_order, fill_value=0)
        pcts = counts / total * 100

        x_positions = np.arange(len(cls_order)) + offset
        axes[1].bar(x_positions, pcts, width=0.35, color=color, alpha=0.75,
                    edgecolor="#333", linewidth=0.5,
                    label=f"{pair_type.replace('-', ' ').title()} (n={total:,})")

    axes[1].set_xticks(range(len(cls_order)))
    axes[1].set_xticklabels(["Strong\n(4-5)", "Moderate\n(3)", "Weak\n(2)", "Fail\n(0-1)"])
    axes[1].set_ylabel("Percentage of Pairs (%)")
    axes[1].set_title("Classification Breakdown by Pair Type")
    axes[1].legend(fontsize=9)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, "intrasector_crosssector.png")
    fig.savefig(out)
    plt.close(fig)
    print(f"  [OK] {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Generating report figures → {OUT_DIR}/\n")

    figures = [
        ("pipeline_diagram", fig_pipeline_diagram),
        ("validation_lift", fig_validation_lift),
        ("backtest_comparison", fig_backtest_comparison),
        ("noise_adjusted_scatter", fig_noise_adjusted_scatter),
        ("pca_variance", fig_pca_variance),
        ("formation_duration_hist", fig_formation_duration),
        ("walkforward_diagram", fig_walkforward_diagram),
        ("frequency_distribution", fig_frequency_distribution),
        ("multi_pnl_grid", fig_multi_pnl_grid),
        ("intrasector_crosssector", fig_intrasector_crosssector),
    ]

    for name, fn in figures:
        print(f"[{name}]")
        try:
            fn()
        except Exception as e:
            import traceback
            print(f"  [ERROR] {e}")
            traceback.print_exc()

    print("\nDone.")


if __name__ == "__main__":
    main()
