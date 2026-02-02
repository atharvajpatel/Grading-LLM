"""
I/O Utilities

Handles all file operations:
- Creating run folders
- Saving responses, embeddings, metrics
- Loading batch inputs
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_logs_dir() -> Path:
    """Get the logs directory, creating it if needed."""
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def create_run_folder(prefix: str = "") -> Path:
    """
    Create a new timestamped run folder.

    Format: logs/YYYYMMDD_HHMMSS_<short_hash>/

    Args:
        prefix: Optional prefix for the folder name

    Returns:
        Path to the created folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create short hash for uniqueness
    hash_input = f"{timestamp}{datetime.now().microsecond}"
    short_hash = hashlib.md5(hash_input.encode()).hexdigest()[:6]

    if prefix:
        folder_name = f"{timestamp}_{prefix}_{short_hash}"
    else:
        folder_name = f"{timestamp}_{short_hash}"

    run_folder = get_logs_dir() / folder_name
    run_folder.mkdir(parents=True, exist_ok=True)

    return run_folder


def save_config(config: Dict[str, Any], path: Path) -> None:
    """
    Save run configuration to JSON.

    Args:
        config: Configuration dictionary
        path: Path to save (folder or file)
    """
    if path.is_dir():
        path = path / "config.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)


def save_responses(responses: List[Dict[str, Any]], path: Path) -> None:
    """
    Save raw API responses to JSONL format.

    Args:
        responses: List of response dictionaries
        path: Path to save (folder or file)
    """
    if path.is_dir():
        path = path / "responses.jsonl"

    with open(path, "w", encoding="utf-8") as f:
        for response in responses:
            f.write(json.dumps(response, default=str) + "\n")


def append_response(response: Dict[str, Any], path: Path) -> None:
    """
    Append a single response to JSONL file.

    Args:
        response: Response dictionary
        path: Path to JSONL file
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(response, default=str) + "\n")


def save_embeddings(embeddings: Dict[str, Any], path: Path) -> None:
    """
    Save embeddings to JSON.

    Args:
        embeddings: Embeddings dictionary
        path: Path to save (folder or file)
    """
    if path.is_dir():
        path = path / "embeddings.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2, default=_numpy_serializer)


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """
    Save metrics to JSON.

    Args:
        metrics: Metrics dictionary
        path: Path to save (folder or file)
    """
    if path.is_dir():
        path = path / "metrics.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=_numpy_serializer)


def load_batch_input(path: Path) -> List[Dict[str, Any]]:
    """
    Load batch input from JSONL file.

    Expected format (each line):
    {"id": "stmt_001", "text": "Statement text"}

    Args:
        path: Path to JSONL file

    Returns:
        List of statement dictionaries
    """
    statements = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            try:
                stmt = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")

            # Validate required fields
            if "text" not in stmt:
                raise ValueError(f"Missing 'text' field on line {line_num}")

            # Auto-generate ID if missing
            if "id" not in stmt:
                stmt["id"] = f"stmt_{line_num:03d}"

            statements.append(stmt)

    return statements


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Generic JSONL loader.

    Args:
        path: Path to JSONL file

    Returns:
        List of dictionaries
    """
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def save_jsonl(items: List[Dict[str, Any]], path: Path) -> None:
    """
    Save list of dicts to JSONL.

    Args:
        items: List of dictionaries
        path: Path to save
    """
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, default=_numpy_serializer) + "\n")


def _numpy_serializer(obj: Any) -> Any:
    """JSON serializer for numpy types."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
