# QUANTT Electron Dashboard

Interactive dashboard for the QUANTT pairs trading strategy and pipeline.

## Project Structure

```
electron-dashboard/
├── public/
│   ├── main.js          # Electron main process
│   ├── preload.js       # Preload script for secure IPC
│   ├── index.html       # Main window HTML
│   └── styles.css       # Global styles
├── src/
│   ├── components/
│   │   ├── Dashboard.js         # Main dashboard component
│   │   ├── Sidebar.js           # Navigation sidebar
│   │   ├── DataUpload.js        # Data upload stage
│   │   ├── PairIdentification.js # Pair identification stage
│   │   ├── StatisticalTests.js  # Statistical tests stage
│   │   └── TradingStrategy.js   # Trading strategy configuration
│   ├── App.js
│   ├── index.js
│   ├── App.css
│   └── index.css
└── package.json
```

## Pipeline Stages

The dashboard implements a 4-stage pipeline:

1. **Data Upload** 📊
   - Import market data (CSV, JSON, Parquet)
   - Validate data format and completeness

2. **Pair Identification** 🔍
   - Identify correlated pairs using DBSCAN/K-Means clustering
   - Calculate correlation matrices

3. **Statistical Tests** 📈
   - Run Augmented Dickey-Fuller (ADF) test
   - Perform Engle-Granger cointegration test
   - Calculate hedge ratios

4. **Trading Strategy** 💹
   - Configure mean reversion parameters
   - Set entrance/exit thresholds and stop loss
   - Run backtest and analyze results

## Getting Started

### Prerequisites

- Node.js 16+
- npm or yarn

### Installation

```bash
cd electron-dashboard
npm install
```

### Development

```bash
npm run dev
```

This will start both the React development server and Electron app.

### Build

```bash
npm run build
```

Creates a production build and packages the Electron app.

## Features

- ✅ Real-time data processing pipeline
- ✅ Interactive pair identification
- ✅ Statistical test integration
- ✅ Strategy parameter configuration
- ✅ Backtest results visualization
- 🔄 Backend integration (in progress)

## Next Steps

- [ ] Connect to Python backend for actual analysis
- [ ] Add charting for price data and spreads
- [ ] Implement live trading connectivity
- [ ] Add historical results database
- [ ] Create trading signals monitor
- [ ] Add performance analytics dashboard
