const fs = require('fs').promises;
const path = require('path');
const Database = require('./database');

class FileSystemManager {
  constructor() {
    this.db = new Database();
  }

  // Ensure user has access to only their workspace
  validateUserPath(userId, requestedPath, userWorkspacePath) {
    const normalizedWorkspace = path.normalize(userWorkspacePath);
    const normalizedRequested = path.normalize(requestedPath);
    
    // Ensure the requested path is within user's workspace
    return normalizedRequested.startsWith(normalizedWorkspace);
  }

  // Get file tree for user's workspace
  async getFileTree(userId, workspaceId, parentPath = null) {
    try {
      const entries = await this.db.getFileEntries(userId, workspaceId, parentPath);
      
      const tree = [];
      for (const entry of entries) {
        const item = {
          id: entry.id,
          name: entry.name,
          path: entry.path,
          type: entry.type,
          size: entry.size,
          permissions: entry.permissions,
          isHidden: entry.is_hidden,
          modifiedAt: entry.modified_at,
          children: entry.type === 'directory' ? [] : undefined
        };

        // If it's a directory, get its children
        if (entry.type === 'directory') {
          item.children = await this.getFileTree(userId, workspaceId, entry.path);
        }

        tree.push(item);
      }

      return tree;
    } catch (error) {
      console.error('Error getting file tree:', error);
      throw error;
    }
  }

  // Scan and sync filesystem with database
  async syncWorkspaceFiles(userId, workspaceId, workspacePath) {
    try {
      await this.scanDirectory(userId, workspaceId, workspacePath, null);
    } catch (error) {
      console.error('Error syncing workspace files:', error);
      throw error;
    }
  }

  async scanDirectory(userId, workspaceId, dirPath, parentPath) {
    try {
      const items = await fs.readdir(dirPath, { withFileTypes: true });
      
      for (const item of items) {
        const itemPath = path.join(dirPath, item.name);
        const relativePath = path.relative(process.cwd(), itemPath);
        
        // Skip hidden files and node_modules by default
        if (item.name.startsWith('.') && item.name !== '.env') continue;
        if (item.name === 'node_modules') continue;

        const stats = await fs.stat(itemPath);
        const type = item.isDirectory() ? 'directory' : 'file';
        
        // Create or update file entry
        const fileEntryId = await this.db.createFileEntry(
          userId,
          workspaceId,
          item.name,
          relativePath,
          parentPath,
          type,
          stats.size
        );

        // If it's a directory, recursively scan it
        if (item.isDirectory()) {
          await this.scanDirectory(userId, workspaceId, itemPath, relativePath);
        }
      }
    } catch (error) {
      console.error('Error scanning directory:', error);
    }
  }

  // File operations
  async createFile(userId, workspaceId, filePath, content = '', userWorkspacePath) {
    try {
      const fullPath = path.join(userWorkspacePath, filePath);
      
      if (!this.validateUserPath(userId, fullPath, userWorkspacePath)) {
        throw new Error('Access denied: Path outside workspace');
      }

      // Ensure directory exists
      await fs.mkdir(path.dirname(fullPath), { recursive: true });
      
      // Create file
      await fs.writeFile(fullPath, content, 'utf8');
      
      // Add to database
      const fileName = path.basename(filePath);
      const parentPath = path.dirname(filePath) === '.' ? null : path.dirname(filePath);
      
      const fileEntryId = await this.db.createFileEntry(
        userId,
        workspaceId,
        fileName,
        filePath,
        parentPath,
        'file',
        Buffer.byteLength(content, 'utf8')
      );

      // Save content to database for small files
      if (content.length < 1024 * 1024) { // 1MB limit
        await this.db.saveFileContent(fileEntryId, content);
      }

      return { success: true, fileEntryId };
    } catch (error) {
      console.error('Error creating file:', error);
      throw error;
    }
  }

  async createDirectory(userId, workspaceId, dirPath, userWorkspacePath) {
    try {
      const fullPath = path.join(userWorkspacePath, dirPath);
      
      if (!this.validateUserPath(userId, fullPath, userWorkspacePath)) {
        throw new Error('Access denied: Path outside workspace');
      }

      // Create directory
      await fs.mkdir(fullPath, { recursive: true });
      
      // Add to database
      const dirName = path.basename(dirPath);
      const parentPath = path.dirname(dirPath) === '.' ? null : path.dirname(dirPath);
      
      const fileEntryId = await this.db.createFileEntry(
        userId,
        workspaceId,
        dirName,
        dirPath,
        parentPath,
        'directory',
        0
      );

      return { success: true, fileEntryId };
    } catch (error) {
      console.error('Error creating directory:', error);
      throw error;
    }
  }

  async readFile(userId, workspaceId, filePath, userWorkspacePath) {
    try {
      const fullPath = path.join(userWorkspacePath, filePath);
      
      if (!this.validateUserPath(userId, fullPath, userWorkspacePath)) {
        throw new Error('Access denied: Path outside workspace');
      }

      // Try to get from database first (for small files)
      const fileEntries = await this.db.getFileEntries(userId, workspaceId);
      const fileEntry = fileEntries.find(entry => entry.path === filePath);
      
      if (fileEntry) {
        await this.db.addRecentFile(userId, fileEntry.id);
        
        const content = await this.db.getFileContent(fileEntry.id);
        if (content) {
          return { content: content.content, encoding: content.encoding };
        }
      }

      // Read from filesystem
      const content = await fs.readFile(fullPath, 'utf8');
      return { content, encoding: 'utf8' };
    } catch (error) {
      console.error('Error reading file:', error);
      throw error;
    }
  }

  async writeFile(userId, workspaceId, filePath, content, userWorkspacePath) {
    try {
      const fullPath = path.join(userWorkspacePath, filePath);
      
      if (!this.validateUserPath(userId, fullPath, userWorkspacePath)) {
        throw new Error('Access denied: Path outside workspace');
      }

      // Write to filesystem
      await fs.writeFile(fullPath, content, 'utf8');
      
      // Update database
      const fileName = path.basename(filePath);
      const parentPath = path.dirname(filePath) === '.' ? null : path.dirname(filePath);
      
      const fileEntryId = await this.db.createFileEntry(
        userId,
        workspaceId,
        fileName,
        filePath,
        parentPath,
        'file',
        Buffer.byteLength(content, 'utf8')
      );

      // Save content to database for small files
      if (content.length < 1024 * 1024) { // 1MB limit
        await this.db.saveFileContent(fileEntryId, content);
      }

      await this.db.addRecentFile(userId, fileEntryId);
      
      return { success: true };
    } catch (error) {
      console.error('Error writing file:', error);
      throw error;
    }
  }

  async deleteFile(userId, workspaceId, filePath, userWorkspacePath) {
    try {
      const fullPath = path.join(userWorkspacePath, filePath);
      
      if (!this.validateUserPath(userId, fullPath, userWorkspacePath)) {
        throw new Error('Access denied: Path outside workspace');
      }

      // Delete from filesystem
      const stats = await fs.stat(fullPath);
      if (stats.isDirectory()) {
        await fs.rmdir(fullPath, { recursive: true });
      } else {
        await fs.unlink(fullPath);
      }
      
      // Delete from database
      await this.db.deleteFileEntry(userId, workspaceId, filePath);
      
      return { success: true };
    } catch (error) {
      console.error('Error deleting file:', error);
      throw error;
    }
  }

  async renameFile(userId, workspaceId, oldPath, newPath, userWorkspacePath) {
    try {
      const fullOldPath = path.join(userWorkspacePath, oldPath);
      const fullNewPath = path.join(userWorkspacePath, newPath);
      
      if (!this.validateUserPath(userId, fullOldPath, userWorkspacePath) ||
          !this.validateUserPath(userId, fullNewPath, userWorkspacePath)) {
        throw new Error('Access denied: Path outside workspace');
      }

      // Rename in filesystem
      await fs.rename(fullOldPath, fullNewPath);
      
      // Update database
      await this.db.deleteFileEntry(userId, workspaceId, oldPath);
      
      const fileName = path.basename(newPath);
      const parentPath = path.dirname(newPath) === '.' ? null : path.dirname(newPath);
      const stats = await fs.stat(fullNewPath);
      
      await this.db.createFileEntry(
        userId,
        workspaceId,
        fileName,
        newPath,
        parentPath,
        stats.isDirectory() ? 'directory' : 'file',
        stats.size
      );
      
      return { success: true };
    } catch (error) {
      console.error('Error renaming file:', error);
      throw error;
    }
  }

  // Search files
  async searchFiles(userId, workspaceId, query, userWorkspacePath) {
    try {
      const entries = await this.db.getFileEntries(userId, workspaceId);
      
      const results = entries.filter(entry => 
        entry.name.toLowerCase().includes(query.toLowerCase()) ||
        entry.path.toLowerCase().includes(query.toLowerCase())
      );

      return results;
    } catch (error) {
      console.error('Error searching files:', error);
      throw error;
    }
  }

  // Get recent files
  async getRecentFiles(userId) {
    try {
      return await this.db.getRecentFiles(userId);
    } catch (error) {
      console.error('Error getting recent files:', error);
      throw error;
    }
  }
}

module.exports = FileSystemManager;
