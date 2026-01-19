import React, { useState } from 'react';
import './StatisticalTests.css';

function StatisticalTests({ pairs, onTestsComplete }) {
  const [loading, setLoading] = useState(false);
  const [testResults, setTestResults] = useState(null);

  const handleRunTests = async () => {
    setLoading(true);

    // Simulate statistical tests
    setTimeout(() => {
      const results = pairs.map(pair => ({
        ...pair,
        adfStatistic: -3.45,
        pValue: 0.008,
        cointegrationScore: 0.92,
        cointegrated: true,
        hedgeRatio: 0.85
      }));
      setTestResults(results);
      onTestsComplete(results);
      setLoading(false);
    }, 2000);
  };

  return (
    <div className="statistical-tests">
      <div className="tests-container">
        <h2>Run Statistical Tests</h2>

        <div className="pairs-list">
          <h3>Identified Pairs</h3>
          {pairs.map((pair, idx) => (
            <div key={idx} className="pair-item">
              <span className="pair-name">{pair.ticker1} / {pair.ticker2}</span>
              <span className="pair-correlation">Corr: {pair.correlation}</span>
            </div>
          ))}
        </div>

        <div className="tests-info">
          <h3>Tests to Run</h3>
          <ul>
            <li><strong>Augmented Dickey-Fuller (ADF):</strong> Check for stationarity</li>
            <li><strong>Engle-Granger Cointegration:</strong> Verify long-term relationship</li>
            <li><strong>Hedge Ratio Calculation:</strong> Optimal weighting for pairs</li>
          </ul>
        </div>

        <button
          className="tests-btn"
          onClick={handleRunTests}
          disabled={loading}
        >
          {loading ? 'Running Tests...' : 'Run Statistical Tests'}
        </button>

        {testResults && (
          <div className="results-summary">
            <h3>Test Results Summary</h3>
            <div className="results-table">
              {testResults.map((result, idx) => (
                <div key={idx} className="result-row">
                  <div className="result-pair">{result.ticker1}/{result.ticker2}</div>
                  <div className="result-status">
                    {result.cointegrated ? '✓ Cointegrated' : '✗ Not Cointegrated'}
                  </div>
                  <div className="result-score">{(result.cointegrationScore * 100).toFixed(1)}%</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default StatisticalTests;
