#!/usr/bin/env python3
"""
Database fix script to recreate the users table with correct schema
"""
import sqlite3
import os

def fix_database():
    """Fix the database schema by recreating the users table"""
    db_path = 'ide_platform.db'
    
    print("🔧 Fixing database schema...")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Drop existing users table if it exists
        print("📋 Dropping existing users table...")
        cursor.execute("DROP TABLE IF EXISTS users")
        
        # Create users table with correct schema
        print("🏗️ Creating users table with correct schema...")
        cursor.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(255) UNIQUE,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                display_name VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create projects table if it doesn't exist
        print("🏗️ Creating projects table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS projects (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name VARCHAR(255) NOT NULL,
                path VARCHAR(500) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create file_entries table if it doesn't exist
        print("🏗️ Creating file_entries table...")
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS file_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                project_id INTEGER,
                file_path VARCHAR(1000) NOT NULL,
                file_type VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
        ''')
        
        # Commit changes
        conn.commit()
        print("✅ Database schema fixed successfully!")
        
        # Verify tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        print(f"📊 Available tables: {[table[0] for table in tables]}")
        
        # Show users table schema
        cursor.execute("PRAGMA table_info(users)")
        columns = cursor.fetchall()
        print("👤 Users table schema:")
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
            
    except Exception as e:
        print(f"❌ Error fixing database: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()

if __name__ == "__main__":
    fix_database()
