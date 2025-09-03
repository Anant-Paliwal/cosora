#!/usr/bin/env python3
"""
Test script to debug registration endpoint issues
"""
import sqlite3
import hashlib
import json
import requests

def test_database():
    """Test database connection and table creation"""
    try:
        print("Testing database connection...")
        conn = sqlite3.connect('ide_platform.db')
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username VARCHAR(255) UNIQUE NOT NULL,
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print("✓ Database table created successfully")
        
        # Test insert
        test_user = {
            'username': 'testuser123',
            'email': 'test123@example.com',
            'password': 'testpass123'
        }
        
        password_hash = hashlib.sha256(test_user['password'].encode('utf-8')).hexdigest()
        
        cursor.execute(
            "INSERT OR REPLACE INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (test_user['username'], test_user['email'], password_hash)
        )
        
        conn.commit()
        print("✓ Test user inserted successfully")
        
        # Verify insert
        cursor.execute("SELECT * FROM users WHERE username = ?", (test_user['username'],))
        result = cursor.fetchone()
        print(f"✓ User retrieved: {result}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Database error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_api_endpoint():
    """Test the registration API endpoint"""
    try:
        print("\nTesting API endpoint...")
        
        test_data = {
            'username': 'apitest456',
            'email': 'apitest456@example.com',
            'password': 'apipass456'
        }
        
        response = requests.post(
            'http://localhost:8000/api/auth/register',
            json=test_data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✓ API registration successful")
            return True
        else:
            print(f"✗ API registration failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ API test error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Registration Debug Test ===")
    
    db_ok = test_database()
    if db_ok:
        api_ok = test_api_endpoint()
        
        if db_ok and api_ok:
            print("\n✓ All tests passed!")
        else:
            print("\n✗ Some tests failed - check the backend server logs")
    else:
        print("\n✗ Database test failed - cannot proceed with API test")
