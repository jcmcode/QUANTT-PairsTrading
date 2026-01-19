import React from 'react';
import './Sidebar.css';

function Sidebar({ activeTab, setActiveTab }) {
  const tabs = [
    { id: 'data', label: 'Data Upload', icon: '📊' },
    { id: 'pairs', label: 'Pair Identification', icon: '🔍' },
    { id: 'tests', label: 'Statistical Tests', icon: '📈' },
    { id: 'strategy', label: 'Trading Strategy', icon: '💹' }
  ];

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <h2>Pipeline</h2>
      </div>
      <nav className="sidebar-nav">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`nav-item ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="icon">{tab.icon}</span>
            <span className="label">{tab.label}</span>
          </button>
        ))}
      </nav>
    </aside>
  );
}

export default Sidebar;
