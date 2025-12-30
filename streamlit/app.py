"""Streamlit UI shell for the pairs trading playground."""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import statsmodels.api as sm
import streamlit as st

# Make sure local modules can be imported when running `streamlit run streamlit/app.py`.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODULE_PATH = PROJECT_ROOT / "trading_part1"
if str(MODULE_PATH) not in sys.path:
    sys.path.append(str(MODULE_PATH))

import data as data_module  # noqa: E402


st.set_page_config(page_title="Pairs Trading Workbench", layout="wide")


@st.cache_data(show_spinner=False)
def load_prices(ticker1: str, ticker2: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Download and clean adjusted close prices for two tickers."""
    return data_module.get_data(
        ticker1, ticker2, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    )


def compute_hedge_ratio(df: pd.DataFrame, ticker1: str, ticker2: str) -> float:
    """Estimate hedge ratio using simple OLS of ticker1 on ticker2."""
    y = df[ticker1]
    x = sm.add_constant(df[ticker2])
    model = sm.OLS(y, x, missing="drop").fit()
    return float(model.params[ticker2])


def build_spread_table(df: pd.DataFrame, ticker1: str, ticker2: str, lookback: int) -> pd.DataFrame:
    """Return spread, rolling mean/std, and z-score for visualization."""
    hedge_ratio = compute_hedge_ratio(df, ticker1, ticker2)
    spread = df[ticker1] - hedge_ratio * df[ticker2]
    rolling_mean = spread.rolling(lookback, min_periods=lookback).mean()
    rolling_std = spread.rolling(lookback, min_periods=lookback).std()
    zscore = (spread - rolling_mean) / rolling_std

    return pd.DataFrame(
        {
            "spread": spread,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "zscore": zscore,
            "hedge_ratio": hedge_ratio,
        }
    )


def price_chart(df: pd.DataFrame) -> alt.Chart:
    price_df = df.reset_index().melt("Date", var_name="Ticker", value_name="Price")
    return (
        alt.Chart(price_df)
        .mark_line()
        .encode(x="Date:T", y="Price:Q", color="Ticker:N")
        .properties(height=320)
    )


def spread_chart(table: pd.DataFrame) -> alt.Chart:
    table = table.reset_index().rename(columns={"index": "Date"})
    base = alt.Chart(table).properties(height=320)

    spread_line = base.mark_line(color="#1f77b4").encode(x="Date:T", y="spread:Q")
    mean_line = base.mark_line(color="#ff7f0e", strokeDash=[6, 4]).encode(
        x="Date:T", y="rolling_mean:Q"
    )

    return spread_line + mean_line


def zscore_chart(table: pd.DataFrame) -> alt.Chart:
    table = table.reset_index().rename(columns={"index": "Date"})
    z_line = (
        alt.Chart(table)
        .mark_line(color="#2ca02c")
        .encode(x="Date:T", y=alt.Y("zscore:Q", title="Z-Score"))
        .properties(height=260)
    )

    thresholds = alt.Data(values=[{"z": -2}, {"z": 0}, {"z": 2}])
    rules = alt.Chart(thresholds).mark_rule(strokeDash=[4, 4], color="#999").encode(y="z:Q")
    return z_line + rules


def render_summary(df: pd.DataFrame, ticker1: str, ticker2: str) -> None:
    start_date = df.index.min().date()
    end_date = df.index.max().date()
    corr = df[ticker1].corr(df[ticker2])

    col1, col2, col3 = st.columns(3)
    col1.metric("Start Date", start_date.isoformat())
    col2.metric("End Date", end_date.isoformat())
    col3.metric("Correlation", f"{corr:.3f}")


def main() -> None:
    st.title("Pairs Trading Workbench")
    st.caption("Interactive sandbox to fetch data, inspect spreads, and iterate on signals.")

    default_end = dt.date.today()
    default_start = default_end - dt.timedelta(days=365 * 2)

    with st.sidebar:
        st.header("Configuration")
        ticker1 = st.text_input("Leg 1 Ticker", value="KO").upper().strip()
        ticker2 = st.text_input("Leg 2 Ticker", value="PEP").upper().strip()
        date_range = st.date_input("Date Range", value=(default_start, default_end))
        lookback = st.slider("Z-Score Lookback (days)", min_value=10, max_value=120, value=30, step=5)
        fetch = st.button("Load Data", type="primary")

    if not fetch:
        st.info("Set tickers and press Load Data to begin.")
        return

    if len(date_range) != 2:
        st.error("Please select a start and end date.")
        return

    start_date, end_date = date_range
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        return

    with st.spinner("Downloading price history..."):
        try:
            df = load_prices(ticker1, ticker2, start_date, end_date)
        except Exception as exc:  # pragma: no cover - handled for UX
            st.error(f"Could not load data: {exc}")
            return

    if df.empty:
        st.warning("No data returned for this configuration.")
        return

    df.index.name = "Date"
    st.subheader("Price History")
    render_summary(df, ticker1, ticker2)
    st.altair_chart(price_chart(df), use_container_width=True)

    with st.spinner("Computing spread and signals..."):
        spread_table = build_spread_table(df, ticker1, ticker2, lookback)

    hedge_ratio = spread_table["hedge_ratio"].iloc[-1]

    st.subheader("Spread & Z-Score")
    st.metric("Hedge Ratio", f"{hedge_ratio:.4f}")

    chart_col, data_col = st.columns([2, 1])
    chart_col.altair_chart(spread_chart(spread_table), use_container_width=True)
    chart_col.altair_chart(zscore_chart(spread_table), use_container_width=True)

    latest = spread_table.dropna().iloc[-1]
    data_col.metric("Latest Spread", f"{latest['spread']:.4f}")
    data_col.metric("Latest Z-Score", f"{latest['zscore']:.2f}")
    data_col.metric("Rolling Std", f"{latest['rolling_std']:.4f}")

    st.subheader("Data Preview")
    preview = pd.concat([df, spread_table[["spread", "rolling_mean", "rolling_std", "zscore"]]], axis=1)
    st.dataframe(preview.tail(200), use_container_width=True)

    st.download_button(
        label="Download CSV",
        data=preview.to_csv().encode("utf-8"),
        file_name=f"{ticker1}_{ticker2}_pairs.csv",
        mime="text/csv",
    )

    st.markdown(
        """
        **Next steps**
        - Hook in the backtest logic once ready and surface performance metrics.
        - Add cointegration checks (ADF, Engle-Granger) using the modules in `part2`.
        - Let users store and recall ticker pairs plus parameter presets.
        """
    )


if __name__ == "__main__":
    main()
