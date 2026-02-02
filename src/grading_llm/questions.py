"""
Question Bank Management

Loads and manages question banks for different modes:
- mech: Mechanistic questions (explicit linguistic features)
- interp: Interpretability questions (implicit meaning, inference)
"""

import json
from pathlib import Path
from typing import Dict, List, Literal, Optional

# Valid question modes
QuestionMode = Literal["mech", "interp"]
VALID_MODES: List[str] = ["mech", "interp"]
DEFAULT_MODE: QuestionMode = "mech"


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data"


def get_questions_path(mode: QuestionMode = DEFAULT_MODE) -> Path:
    """
    Get path to question bank for specified mode.

    Args:
        mode: Question mode ('mech' or 'interp')

    Returns:
        Path to questions JSON file
    """
    if mode not in VALID_MODES:
        raise ValueError(f"Invalid mode '{mode}'. Must be one of: {VALID_MODES}")
    return get_data_dir() / f"questions_{mode}.json"


def get_default_questions_path() -> Path:
    """Get path to default question bank (mech mode for backwards compatibility)."""
    return get_questions_path(DEFAULT_MODE)


def load_questions(path: Optional[Path] = None, mode: QuestionMode = DEFAULT_MODE) -> List[Dict]:
    """
    Load questions from JSON file.

    Args:
        path: Path to questions JSON. If None, uses mode to determine path.
        mode: Question mode ('mech' or 'interp'). Ignored if path is provided.

    Returns:
        List of question dictionaries
    """
    if path is None:
        path = get_questions_path(mode)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("questions", [])


def get_mode_description(mode: QuestionMode) -> str:
    """
    Get human-readable description of a question mode.

    Args:
        mode: Question mode

    Returns:
        Description string
    """
    descriptions = {
        "mech": "Mechanistic - explicit linguistic and semantic features",
        "interp": "Interpretability - implicit meaning, inference, social understanding"
    }
    return descriptions.get(mode, mode)


def get_question_text(question: Dict) -> str:
    """
    Get the question text from a question dictionary.

    Args:
        question: Question dictionary

    Returns:
        Question text string
    """
    return question.get("question", "")


def get_question_by_id(questions: List[Dict], qid: str) -> Optional[Dict]:
    """
    Find a question by its ID.

    Args:
        questions: List of question dictionaries
        qid: Question ID to find

    Returns:
        Question dictionary or None if not found
    """
    for q in questions:
        if q.get("id") == qid:
            return q
    return None


def get_question_by_index(questions: List[Dict], index: int) -> Optional[Dict]:
    """
    Get a question by its index.

    Args:
        questions: List of question dictionaries
        index: 0-based index

    Returns:
        Question dictionary or None if out of range
    """
    if 0 <= index < len(questions):
        return questions[index]
    return None


def get_question_ids(questions: List[Dict]) -> List[str]:
    """Get list of all question IDs."""
    return [q.get("id", f"q_{i}") for i, q in enumerate(questions)]


def get_question_families(questions: List[Dict]) -> List[str]:
    """Get list of all question families (unique)."""
    families = set()
    for q in questions:
        family = q.get("family")
        if family:
            families.add(family)
    return sorted(families)


def questions_to_texts(questions: List[Dict]) -> List[str]:
    """Convert list of question dicts to list of question texts."""
    return [get_question_text(q) for q in questions]
