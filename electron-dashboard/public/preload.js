const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  analyzePairs: (data) => ipcRenderer.invoke('analyze-pairs', data),
  getBacktestResults: (params) => ipcRenderer.invoke('get-backtest-results', params),
  onDataUpdate: (callback) => ipcRenderer.on('data-update', callback),
  removeDataUpdateListener: () => ipcRenderer.removeAllListeners('data-update')
});
