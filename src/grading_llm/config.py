"""
API Key Management - LOCAL DEVELOPMENT ONLY

NOTE: In production, users provide their own OpenAI API key via the frontend.
This config is ONLY used for local development/testing convenience.

The deployed application does NOT use any server-side API keys.
Each user must provide their own OpenAI API key to use the service.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# =============================================================================
# LOCAL DEVELOPMENT ONLY - Paste your key here for local testing
# This is NOT used in production - users provide their own keys via the UI
# =============================================================================
OPENAI_API_KEY = ""
# =============================================================================


def get_api_key_for_local_dev() -> str:
    """
    Get OpenAI API key for LOCAL DEVELOPMENT ONLY.

    In production, users must provide their own API key via the frontend.
    This function is only for convenience during local development/testing.

    Fallback chain:
    1. OPENAI_API_KEY variable in this file
    2. Environment variable OPENAI_API_KEY
    3. .env file in project root

    Returns:
        str: The API key

    Raises:
        ValueError: If no API key is found
    """
    # 1. Check if key is set in this file
    if OPENAI_API_KEY and OPENAI_API_KEY.strip():
        return OPENAI_API_KEY.strip()

    # 2. Load .env file if it exists
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / ".env"
    load_dotenv(env_path)

    # 3. Check environment variable
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if env_key:
        return env_key

    # No key found
    raise ValueError(
        "OpenAI API key not found for local development. Please either:\n"
        "1. Paste your key in src/grading_llm/config.py (OPENAI_API_KEY variable)\n"
        "2. Set the OPENAI_API_KEY environment variable\n"
        "3. Create a .env file in the project root with: OPENAI_API_KEY=your-key"
    )
