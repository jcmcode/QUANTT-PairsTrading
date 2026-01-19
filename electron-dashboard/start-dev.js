#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');

// Start React dev server
const react = spawn('npm', ['run', 'react-start'], {
  cwd: __dirname,
  stdio: 'inherit'
});

// Wait a bit then start Electron
setTimeout(() => {
  process.env.ELECTRON_START_URL = 'http://localhost:3000';
  
  const electron = spawn('electron', ['.'], {
    cwd: __dirname,
    stdio: 'inherit',
    env: { ...process.env, ELECTRON_START_URL: 'http://localhost:3000' }
  });

  electron.on('exit', () => {
    react.kill();
    process.exit();
  });
}, 5000);

process.on('SIGINT', () => {
  react.kill();
  process.exit();
});
