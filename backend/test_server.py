#!/usr/bin/env python3

import uvicorn
from main import app

if __name__ == "__main__":
    print("Starting test server...")
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")