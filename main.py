"""
Backend entry point.

Run with: python main.py

For Render deployment, the PORT env var is used automatically.
"""

import os
import uvicorn
from src.grading_llm.api import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
