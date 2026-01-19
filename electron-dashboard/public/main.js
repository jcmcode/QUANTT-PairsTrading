const { app, BrowserWindow, Menu, ipcMain } = require('electron');
const path = require('path');

// Check if running in development mode
const isDev = !!process.env.ELECTRON_START_URL;

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 800,
    minHeight: 600,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    }
  });

  const startUrl = isDev
    ? process.env.ELECTRON_START_URL
    : `file://${path.join(__dirname, '../build/index.html')}`;

  mainWindow.loadURL(startUrl);

  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (mainWindow === null) {
    createWindow();
  }
});

// IPC handlers for backend communication
ipcMain.handle('analyze-pairs', async (event, data) => {
  try {
    // Placeholder for backend communication
    return { success: true, message: 'Analysis started' };
  } catch (error) {
    return { success: false, error: error.message };
  }
});

ipcMain.handle('get-backtest-results', async (event, params) => {
  try {
    // Placeholder for backend communication
    return { success: true, data: [] };
  } catch (error) {
    return { success: false, error: error.message };
  }
});
