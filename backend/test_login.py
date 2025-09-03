#!/usr/bin/env python3
"""
Test login functionality
"""
import requests
import json

def test_login():
    """Test the login endpoint"""
    url = "http://localhost:8000/api/auth/login"
    
    # Test data
    login_data = {
        "username": "raju@gmail.com",  # Using email as username
        "password": "password123"
    }
    
    try:
        response = requests.post(url, json=login_data)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        
        if response.status_code == 200:
            print("✅ Login successful!")
            return True
        else:
            print("❌ Login failed!")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to server. Make sure the backend is running.")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Login ===")
    test_login()