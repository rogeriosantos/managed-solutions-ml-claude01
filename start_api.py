#!/usr/bin/env python3
"""
Simple script to start the FastAPI server with proper imports.
"""
import sys
import os
import uvicorn

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import after path setup
from api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True, reload_dirs=[project_root])
