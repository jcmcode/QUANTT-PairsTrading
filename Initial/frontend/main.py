"""
QUANTT Pairs Trading Pipeline - Streamlit Dashboard
Main entry point for the multi-page Streamlit application.

This dashboard showcases the end-to-end pairs trading pipeline:
1. Data Ingestion & Exploration
2. Pair Identification (K-Means & DBSCAN Clustering)
3. Cointegration Testing (ADF & Engle-Granger)
4. Backtesting & Trading Execution
5. Performance Metrics & Analysis
"""

import streamlit as st
from streamlit_option_menu import option_menu
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Page configuration
st.set_page_config(
    page_title="QUANTT Pairs Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
    }
    .pipeline-section {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.title("🎯 QUANTT Pipeline")
    
    selected = option_menu(
        menu_title=None,
        options=[
            "Home",
            "Data Exploration",
            "Pair Identification",
            "Cointegration Tests",
            "Backtest Results",
            "Performance Metrics"
        ],
        icons=[
            "house",
            "database",
            "diagram-3",
            "check-circle",
            "graph-up",
            "bar-chart"
        ],
        menu_icon="cast",
        default_index=0,
    )
    
    st.markdown("---")
    st.markdown("""
    ### Pipeline Overview
    This dashboard visualizes a complete pairs trading strategy:
    - **Identify** potential pairs using clustering
    - **Validate** cointegration relationships
    - **Backtest** trading signals
    - **Analyze** performance metrics
    """)

# Route to correct page based on selection
if selected == "Home":
    from pages import home
    home.render()
    
elif selected == "Data Exploration":
    from pages import data_exploration
    data_exploration.render()
    
elif selected == "Pair Identification":
    from pages import pair_identification
    pair_identification.render()
    
elif selected == "Cointegration Tests":
    from pages import cointegration_tests
    cointegration_tests.render()
    
elif selected == "Backtest Results":
    from pages import backtest_results
    backtest_results.render()
    
elif selected == "Performance Metrics":
    from pages import performance_metrics
    performance_metrics.render()
