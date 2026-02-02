"""
Prompt Generation

Creates prompts for the LLM to evaluate statements against questions.
Supports both single-question and batched (20 questions) formats.
"""

from typing import Dict, List

from .scales import format_scale_instruction


def create_single_question_prompt(
    statement: str,
    question: str,
    scale_name: str
) -> str:
    """
    Create a prompt for evaluating a single question.

    Args:
        statement: The statement to evaluate
        question: The question to answer
        scale_name: Name of the grading scale

    Returns:
        Formatted prompt string
    """
    scale_instruction = format_scale_instruction(scale_name)

    return f'''You are analyzing a statement for a specific linguistic feature. Score how DEEPLY and RELEVANTLY the feature appears.

{scale_instruction}

STATEMENT: "{statement}"

Question: {question}

Consider: Is this feature explicit or implicit? Central or peripheral? Strongly or weakly expressed?

Respond with ONLY a JSON object: {{"score": <value>}}'''


def create_batched_prompt(
    statement: str,
    questions: List[Dict],
    scale_name: str,
    batch_size: int = 20
) -> str:
    """
    Create a prompt for evaluating multiple questions at once.

    Args:
        statement: The statement to evaluate
        questions: List of question dictionaries
        scale_name: Name of the grading scale
        batch_size: Maximum questions per batch

    Returns:
        Formatted prompt string
    """
    scale_instruction = format_scale_instruction(scale_name)

    # Build numbered question list
    question_lines = []
    for i, q in enumerate(questions[:batch_size], 1):
        question_text = q.get("question", q) if isinstance(q, dict) else q
        question_lines.append(f"{i}. {question_text}")

    questions_block = "\n".join(question_lines)
    n_questions = len(question_lines)

    return f'''You are analyzing a statement for specific linguistic features. For each question, score how DEEPLY and RELEVANTLY the feature appears in the statement.

{scale_instruction}

STATEMENT TO ANALYZE:
"{statement}"

For each question, consider:
- Is the feature explicitly stated or merely implied?
- Is it central to the statement's meaning or just peripheral/tangential?
- How strongly is it expressed?

Questions:
{questions_block}

Respond with ONLY a JSON object containing an array of scores:
{{"scores": [<score_1>, <score_2>, ..., <score_{n_questions}>]}}'''


def create_binary_evaluation_prompt(
    statement: str,
    question: str
) -> str:
    """
    Create a simple binary evaluation prompt (for pruning).

    Args:
        statement: The statement to evaluate
        question: The question to answer

    Returns:
        Formatted prompt string
    """
    return f'''You are evaluating the following statement:
"{statement}"

Question: {question}

Respond with ONLY a JSON object: {{"answer": 0}} or {{"answer": 1}}

Score must be exactly 0 (no/absent) or 1 (yes/present).'''


def get_expected_response_format(scale_name: str, n_questions: int = 1) -> str:
    """
    Get the expected response format for a given scale.

    Args:
        scale_name: Name of the grading scale
        n_questions: Number of questions (1 for single, >1 for batch)

    Returns:
        Description of expected format
    """
    if n_questions == 1:
        if scale_name == "binary":
            return '{"score": 0} or {"score": 1}'
        elif scale_name == "ternary":
            return '{"score": 0}, {"score": 0.5}, or {"score": 1}'
        elif scale_name == "quaternary":
            return '{"score": 0}, {"score": 0.33}, {"score": 0.66}, or {"score": 1}'
        else:  # continuous
            return '{"score": <decimal between 0.0 and 1.0>}'
    else:
        return f'{{"scores": [<{n_questions} values>]}}'
