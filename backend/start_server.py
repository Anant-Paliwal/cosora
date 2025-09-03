#!/usr/bin/env python3
"""
Simple FastAPI server starter script for Cosora AI IDE Backend
"""

try:
    import uvicorn
    from main import app
    
    print("Starting Cosora AI IDE Backend server...")
    print("Server will be available at: http://localhost:8000")
    print("API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )
    
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Please install required packages:")
    print("pip install fastapi uvicorn python-multipart websockets google-generativeai bcrypt PyJWT python-dotenv")
    
except Exception as e:
    print(f"Error starting server: {e}")
    import traceback
    traceback.print_exc()
