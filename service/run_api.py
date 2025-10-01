#!/usr/bin/env python3
"""
Startup script for the Stock Prediction API
Run this to start the FastAPI server
"""

import uvicorn
import os
import sys

if __name__ == "__main__":
    # Change to the service directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print("Starting Stock Prediction API...")
    print(f"Working directory: {os.getcwd()}")
    print("API will be available at: http://localhost:8000")
    print("API documentation: http://localhost:8000/docs")
    print("Health check: http://localhost:8000/health")
    
    # Start the server
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Accept connections from any IP
        port=8000,
        reload=True,
        log_level="info"
    )
