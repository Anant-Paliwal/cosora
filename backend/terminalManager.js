const pty = require('node-pty');
const WebSocket = require('ws');
const os = require('os');
const path = require('path');

class TerminalManager {
  constructor() {
    this.terminals = new Map();
    this.clients = new Map();
  }

  createTerminal(sessionId, workingDir = process.cwd()) {
    const shell = os.platform() === 'win32' ? 'powershell.exe' : 'bash';
    
    const terminal = pty.spawn(shell, [], {
      name: 'xterm-color',
      cols: 80,
      rows: 24,
      cwd: workingDir,
      env: process.env
    });

    this.terminals.set(sessionId, {
      terminal,
      workingDir,
      createdAt: new Date(),
      lastActivity: new Date()
    });

    // Handle terminal output
    terminal.on('data', (data) => {
      this.broadcastToClients(sessionId, {
        type: 'output',
        data: data
      });
    });

    terminal.on('exit', (code) => {
      this.broadcastToClients(sessionId, {
        type: 'exit',
        code: code
      });
      this.terminals.delete(sessionId);
    });

    return terminal;
  }

  writeToTerminal(sessionId, data) {
    const terminalSession = this.terminals.get(sessionId);
    if (terminalSession) {
      terminalSession.terminal.write(data);
      terminalSession.lastActivity = new Date();
      return true;
    }
    return false;
  }

  resizeTerminal(sessionId, cols, rows) {
    const terminalSession = this.terminals.get(sessionId);
    if (terminalSession) {
      terminalSession.terminal.resize(cols, rows);
      return true;
    }
    return false;
  }

  killTerminal(sessionId) {
    const terminalSession = this.terminals.get(sessionId);
    if (terminalSession) {
      terminalSession.terminal.kill();
      this.terminals.delete(sessionId);
      return true;
    }
    return false;
  }

  addClient(sessionId, ws) {
    if (!this.clients.has(sessionId)) {
      this.clients.set(sessionId, new Set());
    }
    this.clients.get(sessionId).add(ws);

    ws.on('close', () => {
      this.removeClient(sessionId, ws);
    });
  }

  removeClient(sessionId, ws) {
    const clients = this.clients.get(sessionId);
    if (clients) {
      clients.delete(ws);
      if (clients.size === 0) {
        this.clients.delete(sessionId);
      }
    }
  }

  broadcastToClients(sessionId, message) {
    const clients = this.clients.get(sessionId);
    if (clients) {
      const messageStr = JSON.stringify(message);
      clients.forEach(ws => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(messageStr);
        }
      });
    }
  }

  getTerminalSessions() {
    const sessions = [];
    this.terminals.forEach((session, sessionId) => {
      sessions.push({
        id: sessionId,
        workingDir: session.workingDir,
        createdAt: session.createdAt,
        lastActivity: session.lastActivity,
        isActive: true
      });
    });
    return sessions;
  }

  // Clean up inactive terminals
  cleanupInactiveTerminals(maxIdleTime = 30 * 60 * 1000) { // 30 minutes
    const now = new Date();
    this.terminals.forEach((session, sessionId) => {
      if (now - session.lastActivity > maxIdleTime) {
        this.killTerminal(sessionId);
      }
    });
  }
}

module.exports = TerminalManager;
