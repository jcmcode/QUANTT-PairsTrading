import React, { useState } from 'react';
import './PairIdentification.css';

function PairIdentification({ data, onPairsIdentified }) {
  const [loading, setLoading] = useState(false);
  const [method, setMethod] = useState('dbscan');

  const handleIdentify = async () => {
    setLoading(true);

    // Simulate pair identification
    setTimeout(() => {
      const mockPairs = [
        { ticker1: 'AAPL', ticker2: 'MSFT', correlation: 0.87, distance: 0.23 },
        { ticker1: 'GOOGL', ticker2: 'MSFT', correlation: 0.81, distance: 0.35 },
        { ticker1: 'AMZN', ticker2: 'NVDA', correlation: 0.76, distance: 0.42 }
      ];
      onPairsIdentified(mockPairs);
      setLoading(false);
    }, 2000);
  };

  return (
    <div className="pair-identification">
      <div className="identification-container">
        <h2>Identify Correlated Pairs</h2>

        <div className="method-selector">
          <label>Identification Method:</label>
          <select value={method} onChange={(e) => setMethod(e.target.value)} disabled={loading}>
            <option value="dbscan">DBSCAN Clustering</option>
            <option value="kmeans">K-Means Clustering</option>
            <option value="correlation">Correlation Matrix</option>
          </select>
        </div>

        <div className="data-info">
          <p><strong>Loaded Data:</strong> {data.file}</p>
          <p><strong>Tickers:</strong> {data.tickers.join(', ')}</p>
          <p><strong>Date Range:</strong> {data.dateRange}</p>
          <p><strong>Data Points:</strong> {data.rows}</p>
        </div>

        <button
          className="identify-btn"
          onClick={handleIdentify}
          disabled={loading}
        >
          {loading ? 'Identifying Pairs...' : 'Identify Pairs'}
        </button>
      </div>
    </div>
  );
}

export default PairIdentification;
