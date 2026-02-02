"""
Question Generation

Defines factor families and provides utilities for generating
candidate questions with minimal pairs.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

# 20 Factor Families - semantic axes designed to be roughly independent
FACTOR_FAMILIES = [
    "named_entities",   # Named entities present
    "actions_events",   # Actions/events present
    "causality",        # Causality language
    "temporal",         # Temporal reference
    "spatial",          # Spatial/location reference
    "numeric",          # Numeric quantity reference
    "negation",         # Negation
    "uncertainty",      # Uncertainty/hedging
    "modality",         # Should/could/must
    "sentiment",        # Sentiment/valence
    "emotion",          # Explicit emotion words
    "social",           # Social interaction
    "dialogue",         # Dialogue/quoted speech
    "first_person",     # First-person perspective
    "imperative",       # Instruction/imperative
    "comparison",       # Comparison
    "normative",        # Normative/judgment language
    "intent",           # Intent/goal
    "concreteness",     # Concrete vs abstract
    "identity",         # Identity/self-description
]

# Family descriptions for documentation
FAMILY_DESCRIPTIONS = {
    "named_entities": "Questions about specific named entities (people, places, organizations)",
    "actions_events": "Questions about actions, events, processes, and motion",
    "causality": "Questions about cause-effect relationships and explanations",
    "temporal": "Questions about time references, ordering, and duration",
    "spatial": "Questions about locations, positions, and directions",
    "numeric": "Questions about numbers, quantities, and measurements",
    "negation": "Questions about negation, denial, and absence",
    "uncertainty": "Questions about hedging, doubt, and probabilistic language",
    "modality": "Questions about obligation, possibility, and necessity",
    "sentiment": "Questions about positive/negative sentiment and evaluation",
    "emotion": "Questions about explicit emotions and emotional language",
    "social": "Questions about social interaction and relationships",
    "dialogue": "Questions about quoted speech and verbal communication",
    "first_person": "Questions about first-person perspective and personal expression",
    "imperative": "Questions about commands, instructions, and directives",
    "comparison": "Questions about comparisons, similarities, and differences",
    "normative": "Questions about moral judgments and value statements",
    "intent": "Questions about intentions, goals, and purposes",
    "concreteness": "Questions about concrete vs abstract content",
    "identity": "Questions about identity, roles, and characteristics",
}


def get_data_dir() -> Path:
    """Get the data directory path."""
    return Path(__file__).parent.parent.parent / "data"


def load_candidate_questions(path: Optional[Path] = None) -> List[Dict]:
    """
    Load candidate questions from JSON file.

    Args:
        path: Path to questions JSON. Default: data/questions_100.json

    Returns:
        List of question dictionaries
    """
    if path is None:
        path = get_data_dir() / "questions_100.json"

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data.get("questions", [])


def get_questions_by_family(
    questions: List[Dict],
    family: str
) -> List[Dict]:
    """
    Filter questions by family.

    Args:
        questions: List of question dictionaries
        family: Family name to filter by

    Returns:
        Filtered list of questions
    """
    return [q for q in questions if q.get("family") == family]


def get_family_count(questions: List[Dict]) -> Dict[str, int]:
    """
    Count questions per family.

    Args:
        questions: List of question dictionaries

    Returns:
        Dictionary mapping family name to count
    """
    counts = {}
    for q in questions:
        family = q.get("family", "unknown")
        counts[family] = counts.get(family, 0) + 1
    return counts


def validate_question_format(question: Dict) -> List[str]:
    """
    Validate that a question has all required fields.

    Args:
        question: Question dictionary

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    required_fields = ["id", "family", "question"]

    for field in required_fields:
        if field not in question:
            errors.append(f"Missing required field: {field}")

    if "family" in question and question["family"] not in FACTOR_FAMILIES:
        errors.append(f"Unknown family: {question['family']}")

    if "minimal_pairs" in question:
        mp = question["minimal_pairs"]
        if "positive" not in mp:
            errors.append("minimal_pairs missing 'positive' example")
        if "negative" not in mp:
            errors.append("minimal_pairs missing 'negative' example")

    return errors


def validate_question_bank(questions: List[Dict]) -> Dict[str, List[str]]:
    """
    Validate all questions in a bank.

    Args:
        questions: List of question dictionaries

    Returns:
        Dictionary mapping question ID to list of errors
    """
    all_errors = {}
    seen_ids = set()

    for q in questions:
        qid = q.get("id", "unknown")

        # Check for duplicate IDs
        if qid in seen_ids:
            all_errors.setdefault(qid, []).append("Duplicate question ID")
        seen_ids.add(qid)

        # Validate format
        errors = validate_question_format(q)
        if errors:
            all_errors.setdefault(qid, []).extend(errors)

    return all_errors


def get_minimal_pair(question: Dict) -> Optional[Dict[str, str]]:
    """
    Get the minimal pair for a question.

    Args:
        question: Question dictionary

    Returns:
        Minimal pair dict with 'positive' and 'negative' keys, or None
    """
    return question.get("minimal_pairs")
