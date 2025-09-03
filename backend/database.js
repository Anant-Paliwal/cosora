const sqlite3 = require('sqlite3').verbose();
const path = require('path');
const fs = require('fs');

class Database {
  constructor() {
    this.dbPath = path.join(__dirname, 'ide_platform.db');
    this.db = null;
    this.init();
  }

  init() {
    this.db = new sqlite3.Database(this.dbPath, (err) => {
      if (err) {
        console.error('Error opening database:', err);
      } else {
        console.log('Connected to SQLite database');
        this.createTables();
      }
    });
  }

  createTables() {
    const tables = [
      // Users table
      `CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username VARCHAR(255) UNIQUE NOT NULL,
        email VARCHAR(255) UNIQUE NOT NULL,
        password_hash VARCHAR(255) NOT NULL,
        workspace_path VARCHAR(500),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,

      // User workspaces - each user gets isolated workspace
      `CREATE TABLE IF NOT EXISTS user_workspaces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        name VARCHAR(255) NOT NULL,
        path VARCHAR(500) NOT NULL,
        is_active BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
      )`,

      // File system entries for each user
      `CREATE TABLE IF NOT EXISTS file_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        workspace_id INTEGER NOT NULL,
        name VARCHAR(255) NOT NULL,
        path VARCHAR(1000) NOT NULL,
        parent_path VARCHAR(1000),
        type VARCHAR(20) NOT NULL, -- 'file' or 'directory'
        size INTEGER DEFAULT 0,
        permissions VARCHAR(10) DEFAULT 'rw-r--r--',
        is_hidden BOOLEAN DEFAULT 0,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        modified_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (workspace_id) REFERENCES user_workspaces (id) ON DELETE CASCADE,
        UNIQUE(user_id, workspace_id, path)
      )`,

      // File content storage (for small files)
      `CREATE TABLE IF NOT EXISTS file_contents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        file_entry_id INTEGER NOT NULL,
        content TEXT,
        encoding VARCHAR(20) DEFAULT 'utf8',
        hash VARCHAR(64),
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (file_entry_id) REFERENCES file_entries (id) ON DELETE CASCADE
      )`,

      // Terminal sessions for each user
      `CREATE TABLE IF NOT EXISTS terminal_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        workspace_id INTEGER NOT NULL,
        session_id VARCHAR(255) UNIQUE NOT NULL,
        working_directory VARCHAR(500),
        is_active BOOLEAN DEFAULT 1,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (workspace_id) REFERENCES user_workspaces (id) ON DELETE CASCADE
      )`,

      // Recent files for quick access
      `CREATE TABLE IF NOT EXISTS recent_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        file_entry_id INTEGER NOT NULL,
        accessed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
        FOREIGN KEY (file_entry_id) REFERENCES file_entries (id) ON DELETE CASCADE
      )`
    ];

    tables.forEach(tableSQL => {
      this.db.run(tableSQL, (err) => {
        if (err) {
          console.error('Error creating table:', err);
        }
      });
    });

    // Create indexes for better performance
    const indexes = [
      'CREATE INDEX IF NOT EXISTS idx_file_entries_user_workspace ON file_entries(user_id, workspace_id)',
      'CREATE INDEX IF NOT EXISTS idx_file_entries_parent_path ON file_entries(parent_path)',
      'CREATE INDEX IF NOT EXISTS idx_terminal_sessions_user ON terminal_sessions(user_id, is_active)',
      'CREATE INDEX IF NOT EXISTS idx_recent_files_user ON recent_files(user_id, accessed_at DESC)'
    ];

    indexes.forEach(indexSQL => {
      this.db.run(indexSQL);
    });
  }

  // User management
  async createUser(username, email, passwordHash, workspacePath) {
    return new Promise((resolve, reject) => {
      const stmt = this.db.prepare(`
        INSERT INTO users (username, email, password_hash, workspace_path) 
        VALUES (?, ?, ?, ?)
      `);
      
      stmt.run([username, email, passwordHash, workspacePath], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.lastID);
        }
      });
      stmt.finalize();
    });
  }

  async getUserById(userId) {
    return new Promise((resolve, reject) => {
      this.db.get('SELECT * FROM users WHERE id = ?', [userId], (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }

  async getUserByUsername(username) {
    return new Promise((resolve, reject) => {
      this.db.get('SELECT * FROM users WHERE username = ?', [username], (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }

  async getUserByEmail(email) {
    return new Promise((resolve, reject) => {
      this.db.get('SELECT * FROM users WHERE email = ?', [email], (err, row) => {
        if (err) reject(err);
        else resolve(row);
      });
    });
  }

  // Workspace management
  async createWorkspace(userId, name, path) {
    return new Promise((resolve, reject) => {
      const stmt = this.db.prepare(`
        INSERT INTO user_workspaces (user_id, name, path, is_active) 
        VALUES (?, ?, ?, 1)
      `);
      
      stmt.run([userId, name, path], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.lastID);
        }
      });
      stmt.finalize();
    });
  }

  async getUserWorkspaces(userId) {
    return new Promise((resolve, reject) => {
      this.db.all(
        'SELECT * FROM user_workspaces WHERE user_id = ? ORDER BY created_at DESC',
        [userId],
        (err, rows) => {
          if (err) reject(err);
          else resolve(rows);
        }
      );
    });
  }

  // File system operations
  async createFileEntry(userId, workspaceId, name, path, parentPath, type, size = 0) {
    return new Promise((resolve, reject) => {
      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO file_entries 
        (user_id, workspace_id, name, path, parent_path, type, size, modified_at) 
        VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
      `);
      
      stmt.run([userId, workspaceId, name, path, parentPath, type, size], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.lastID);
        }
      });
      stmt.finalize();
    });
  }

  async getFileEntries(userId, workspaceId, parentPath = null) {
    return new Promise((resolve, reject) => {
      const query = parentPath 
        ? 'SELECT * FROM file_entries WHERE user_id = ? AND workspace_id = ? AND parent_path = ? ORDER BY type DESC, name ASC'
        : 'SELECT * FROM file_entries WHERE user_id = ? AND workspace_id = ? AND parent_path IS NULL ORDER BY type DESC, name ASC';
      
      const params = parentPath ? [userId, workspaceId, parentPath] : [userId, workspaceId];
      
      this.db.all(query, params, (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  async deleteFileEntry(userId, workspaceId, path) {
    return new Promise((resolve, reject) => {
      this.db.run(
        'DELETE FROM file_entries WHERE user_id = ? AND workspace_id = ? AND (path = ? OR path LIKE ?)',
        [userId, workspaceId, path, path + '/%'],
        function(err) {
          if (err) reject(err);
          else resolve(this.changes);
        }
      );
    });
  }

  // File content operations
  async saveFileContent(fileEntryId, content, encoding = 'utf8') {
    const crypto = require('crypto');
    const hash = crypto.createHash('sha256').update(content).digest('hex');
    
    return new Promise((resolve, reject) => {
      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO file_contents (file_entry_id, content, encoding, hash) 
        VALUES (?, ?, ?, ?)
      `);
      
      stmt.run([fileEntryId, content, encoding, hash], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.lastID);
        }
      });
      stmt.finalize();
    });
  }

  async getFileContent(fileEntryId) {
    return new Promise((resolve, reject) => {
      this.db.get(
        'SELECT * FROM file_contents WHERE file_entry_id = ? ORDER BY created_at DESC LIMIT 1',
        [fileEntryId],
        (err, row) => {
          if (err) reject(err);
          else resolve(row);
        }
      );
    });
  }

  // Terminal session management
  async createTerminalSession(userId, workspaceId, sessionId, workingDirectory) {
    return new Promise((resolve, reject) => {
      const stmt = this.db.prepare(`
        INSERT INTO terminal_sessions (user_id, workspace_id, session_id, working_directory) 
        VALUES (?, ?, ?, ?)
      `);
      
      stmt.run([userId, workspaceId, sessionId, workingDirectory], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.lastID);
        }
      });
      stmt.finalize();
    });
  }

  async getActiveTerminalSessions(userId, workspaceId) {
    return new Promise((resolve, reject) => {
      this.db.all(
        'SELECT * FROM terminal_sessions WHERE user_id = ? AND workspace_id = ? AND is_active = 1',
        [userId, workspaceId],
        (err, rows) => {
          if (err) reject(err);
          else resolve(rows);
        }
      );
    });
  }

  async updateTerminalActivity(sessionId) {
    return new Promise((resolve, reject) => {
      this.db.run(
        'UPDATE terminal_sessions SET last_activity = CURRENT_TIMESTAMP WHERE session_id = ?',
        [sessionId],
        function(err) {
          if (err) reject(err);
          else resolve(this.changes);
        }
      );
    });
  }

  // Recent files tracking
  async addRecentFile(userId, fileEntryId) {
    return new Promise((resolve, reject) => {
      const stmt = this.db.prepare(`
        INSERT OR REPLACE INTO recent_files (user_id, file_entry_id, accessed_at) 
        VALUES (?, ?, CURRENT_TIMESTAMP)
      `);
      
      stmt.run([userId, fileEntryId], function(err) {
        if (err) {
          reject(err);
        } else {
          resolve(this.lastID);
        }
      });
      stmt.finalize();
    });
  }

  async getRecentFiles(userId, limit = 10) {
    return new Promise((resolve, reject) => {
      this.db.all(`
        SELECT fe.*, rf.accessed_at 
        FROM recent_files rf 
        JOIN file_entries fe ON rf.file_entry_id = fe.id 
        WHERE rf.user_id = ? 
        ORDER BY rf.accessed_at DESC 
        LIMIT ?
      `, [userId, limit], (err, rows) => {
        if (err) reject(err);
        else resolve(rows);
      });
    });
  }

  close() {
    if (this.db) {
      this.db.close((err) => {
        if (err) {
          console.error('Error closing database:', err);
        } else {
          console.log('Database connection closed');
        }
      });
    }
  }
}

module.exports = Database;
