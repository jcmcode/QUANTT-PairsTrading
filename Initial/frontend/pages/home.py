"""
Home Page - Pipeline Overview
Showcases the overall project goals and architecture.
"""

import streamlit as st


def render():
    """Render the home page with pipeline overview."""
    
    st.markdown("""
    <div class='main-header'>
    <h1>📈 QUANTT Pairs Trading Pipeline</h1>
    <p style='font-size: 1.2rem; color: #666;'>
    End-to-end pairs trading strategy discovery and backtesting
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Pipeline Overview
    st.markdown("## 🎯 Pipeline Architecture")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown("""
        <div class='pipeline-section'>
        <h3 style='text-align: center;'>1️⃣ Data</h3>
        <p style='text-align: center; font-size: 0.9rem;'>
        Ingest & store historical price data
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='pipeline-section'>
        <h3 style='text-align: center;'>2️⃣ Identify</h3>
        <p style='text-align: center; font-size: 0.9rem;'>
        Cluster assets (K-Means, DBSCAN)
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='pipeline-section'>
        <h3 style='text-align: center;'>3️⃣ Validate</h3>
        <p style='text-align: center; font-size: 0.9rem;'>
        Test cointegration (ADF, EG)
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class='pipeline-section'>
        <h3 style='text-align: center;'>4️⃣ Trade</h3>
        <p style='text-align: center; font-size: 0.9rem;'>
        Generate z-score signals
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown("""
        <div class='pipeline-section'>
        <h3 style='text-align: center;'>5️⃣ Analyze</h3>
        <p style='text-align: center; font-size: 0.9rem;'>
        Performance metrics & stats
        </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Key Concepts
    st.markdown("## 💡 Key Concepts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Pairs Trading Strategy
        A market-neutral strategy that:
        - Identifies pairs of correlated assets
        - Exploits temporary price divergence (mean reversion)
        - Takes long/short positions on the spread
        - Generates signals based on statistical deviations
        
        **Mean-reversion** assumption: when the spread widens beyond historical norms,
        we expect it to revert to the mean.
        """)
    
    with col2:
        st.markdown("""
        ### Cointegration Testing
        Validates that asset pairs move together long-term:
        - **ADF Test**: Tests for stationarity of the spread
        - **Engle-Granger Test**: Tests for cointegration relationship
        
        **Why it matters**: Without cointegration, the spread
        may diverge indefinitely, leading to massive losses.
        """)
    
    st.markdown("---")
    
    # Dashboard Features
    st.markdown("## 📊 Dashboard Features")
    
    features = {
        "📁 Data Exploration": "Explore raw data, view tickers, summary statistics",
        "🎯 Pair Identification": "Visualize K-Means and DBSCAN clustering results, identify candidate pairs",
        "✅ Cointegration Tests": "View ADF and Engle-Granger test results, p-values, and validation metrics",
        "📈 Backtest Results": "Equity curves, entry/exit markers on price charts, trade execution details",
        "📊 Performance Metrics": "Sharpe ratio, maximum drawdown, P/L, win rate, trade statistics"
    }
    
    for feature, description in features.items():
        st.markdown(f"**{feature}**: {description}")
    
    st.markdown("---")
    
    # Getting Started
    st.markdown("## 🚀 Getting Started")
    
    st.info("""
    Use the sidebar navigation to explore different sections of the pipeline.
    Each section builds on the previous one, showcasing how raw data becomes a trading strategy.
    """)
