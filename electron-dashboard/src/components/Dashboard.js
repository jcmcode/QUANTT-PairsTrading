import React, { useState } from 'react';
import './Dashboard.css';
import DataUpload from './DataUpload';
import PairIdentification from './PairIdentification';
import StatisticalTests from './StatisticalTests';
import TradingStrategy from './TradingStrategy';
import Sidebar from './Sidebar';

function Dashboard() {
  const [activeTab, setActiveTab] = useState('data');
  const [pipelineData, setPipelineData] = useState({
    data: null,
    pairs: null,
    cointegration: null,
    backtest: null
  });

  const handleDataLoaded = (data) => {
    setPipelineData(prev => ({ ...prev, data }));
    setActiveTab('pairs');
  };

  const handlePairsIdentified = (pairs) => {
    setPipelineData(prev => ({ ...prev, pairs }));
    setActiveTab('tests');
  };

  const handleTestsComplete = (cointegration) => {
    setPipelineData(prev => ({ ...prev, cointegration }));
    setActiveTab('strategy');
  };

  const handleStrategyComplete = (backtest) => {
    setPipelineData(prev => ({ ...prev, backtest }));
  };

  return (
    <div className="dashboard">
      <Sidebar activeTab={activeTab} setActiveTab={setActiveTab} />
      <main className="dashboard-content">
        <header className="dashboard-header">
          <h1>QUANTT Pairs Trading Dashboard</h1>
          <p>Data → Identifiers → Statistical Tests → Trading Strategy</p>
        </header>

        <div className="pipeline-tabs">
          {activeTab === 'data' && (
            <DataUpload onDataLoaded={handleDataLoaded} />
          )}
          {activeTab === 'pairs' && pipelineData.data && (
            <PairIdentification data={pipelineData.data} onPairsIdentified={handlePairsIdentified} />
          )}
          {activeTab === 'tests' && pipelineData.pairs && (
            <StatisticalTests pairs={pipelineData.pairs} onTestsComplete={handleTestsComplete} />
          )}
          {activeTab === 'strategy' && pipelineData.cointegration && (
            <TradingStrategy cointegration={pipelineData.cointegration} onStrategyComplete={handleStrategyComplete} />
          )}
        </div>

        {pipelineData.backtest && (
          <section className="backtest-results">
            <h2>Backtest Results</h2>
            <div className="results-grid">
              <div className="result-card">
                <h3>Total Return</h3>
                <p className="metric">{pipelineData.backtest.totalReturn || 'N/A'}</p>
              </div>
              <div className="result-card">
                <h3>Sharpe Ratio</h3>
                <p className="metric">{pipelineData.backtest.sharpeRatio || 'N/A'}</p>
              </div>
              <div className="result-card">
                <h3>Max Drawdown</h3>
                <p className="metric">{pipelineData.backtest.maxDrawdown || 'N/A'}</p>
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default Dashboard;
