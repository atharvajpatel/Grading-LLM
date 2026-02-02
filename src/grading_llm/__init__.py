"""
QA-Embedding Granularity Experiment Harness

Measure LLM consistency when generating interpretable "question-answering embeddings"
across different answer granularities (binary, ternary, quaternary, continuous).
"""

__version__ = "0.1.0"

from .config import get_api_key_for_local_dev
from .scales import SCALES, SCALE_ORDER, validate_response, format_scale_instruction
from .questions import load_questions, get_question_text

__all__ = [
    "get_api_key_for_local_dev",
    "SCALES",
    "SCALE_ORDER",
    "validate_response",
    "format_scale_instruction",
    "load_questions",
    "get_question_text",
]
