import React, { useState } from 'react';
import './DataUpload.css';

function DataUpload({ onDataLoaded }) {
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState('');

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (file) {
      setLoading(true);
      setFileName(file.name);

      // Simulate file upload and processing
      setTimeout(() => {
        const mockData = {
          file: file.name,
          rows: 1000,
          tickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA'],
          dateRange: '2023-01-01 to 2024-01-19',
          loaded: true
        };
        onDataLoaded(mockData);
        setLoading(false);
      }, 2000);
    }
  };

  return (
    <div className="data-upload">
      <div className="upload-container">
        <h2>Upload Market Data</h2>
        <div className="upload-zone">
          <input
            type="file"
            id="file-input"
            onChange={handleFileUpload}
            accept=".csv,.json,.parquet"
            disabled={loading}
          />
          <label htmlFor="file-input" className={`upload-label ${loading ? 'loading' : ''}`}>
            {loading ? (
              <>
                <span className="spinner"></span>
                Processing {fileName}...
              </>
            ) : (
              <>
                <span className="upload-icon">📁</span>
                <span>Drag and drop CSV/JSON or click to select</span>
                <span className="file-formats">Supported: .csv, .json, .parquet</span>
              </>
            )}
          </label>
        </div>

        <div className="upload-info">
          <h3>Data Requirements</h3>
          <ul>
            <li>Time series data with date and OHLCV (Open, High, Low, Close, Volume)</li>
            <li>Multiple ticker symbols for pair identification</li>
            <li>Regular time intervals (daily, hourly, etc.)</li>
            <li>At least 252 data points (1 year of daily data)</li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default DataUpload;
