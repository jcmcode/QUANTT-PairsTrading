import React, { useState } from 'react';
import './TradingStrategy.css';

function TradingStrategy({ cointegration, onStrategyComplete }) {
  const [loading, setLoading] = useState(false);
  const [strategyParams, setStrategyParams] = useState({
    entranceThreshold: 2.0,
    exitThreshold: 0.5,
    stopLoss: 3.0,
    position_size: 0.1
  });

  const handleParamChange = (param, value) => {
    setStrategyParams(prev => ({ ...prev, [param]: parseFloat(value) }));
  };

  const handleBacktest = async () => {
    setLoading(true);

    // Simulate backtesting
    setTimeout(() => {
      const results = {
        totalReturn: '18.5%',
        sharpeRatio: 1.85,
        maxDrawdown: '-8.2%',
        winRate: '62.3%',
        trades: 127,
        parameters: strategyParams
      };
      onStrategyComplete(results);
      setLoading(false);
    }, 3000);
  };

  return (
    <div className="trading-strategy">
      <div className="strategy-container">
        <h2>Configure & Backtest Strategy</h2>

        <div className="strategy-grid">
          <div className="params-section">
            <h3>Strategy Parameters</h3>

            <div className="param-control">
              <label>Entrance Threshold (σ)</label>
              <input
                type="number"
                step="0.1"
                value={strategyParams.entranceThreshold}
                onChange={(e) => handleParamChange('entranceThreshold', e.target.value)}
                disabled={loading}
              />
              <span className="param-hint">Standard deviations for entry signal</span>
            </div>

            <div className="param-control">
              <label>Exit Threshold (σ)</label>
              <input
                type="number"
                step="0.1"
                value={strategyParams.exitThreshold}
                onChange={(e) => handleParamChange('exitThreshold', e.target.value)}
                disabled={loading}
              />
              <span className="param-hint">Standard deviations for exit signal</span>
            </div>

            <div className="param-control">
              <label>Stop Loss (σ)</label>
              <input
                type="number"
                step="0.1"
                value={strategyParams.stopLoss}
                onChange={(e) => handleParamChange('stopLoss', e.target.value)}
                disabled={loading}
              />
              <span className="param-hint">Maximum loss before position closure</span>
            </div>

            <div className="param-control">
              <label>Position Size</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={strategyParams.position_size}
                onChange={(e) => handleParamChange('position_size', e.target.value)}
                disabled={loading}
              />
              <span className="param-hint">Fraction of portfolio per trade</span>
            </div>
          </div>

          <div className="strategy-info">
            <h3>Strategy Overview</h3>
            <div className="info-box">
              <h4>Mean Reversion Strategy</h4>
              <p>
                When the spread between cointegrated pairs exceeds entrance threshold,
                a mean reversion signal is generated. The strategy assumes the spread
                will revert to its historical mean.
              </p>
              <ul>
                <li><strong>Signal:</strong> Z-score based on historical mean and std dev</li>
                <li><strong>Entry:</strong> When |Z-score| > entrance threshold</li>
                <li><strong>Exit:</strong> When |Z-score| &lt; exit threshold or stop loss hit</li>
                <li><strong>Hedging:</strong> Long/short positions weighted by hedge ratio</li>
              </ul>
            </div>
          </div>
        </div>

        <button
          className="backtest-btn"
          onClick={handleBacktest}
          disabled={loading}
        >
          {loading ? 'Running Backtest...' : 'Run Backtest'}
        </button>
      </div>
    </div>
  );
}

export default TradingStrategy;
