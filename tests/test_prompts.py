"""Tests for prompts module."""

import pytest

from grading_llm.prompts import (
    create_single_question_prompt,
    create_batched_prompt,
    create_binary_evaluation_prompt,
    get_expected_response_format,
)


class TestSingleQuestionPrompt:
    def test_contains_statement(self):
        """Prompt should contain the statement."""
        prompt = create_single_question_prompt(
            "Test statement", "Test question?", "binary"
        )
        assert "Test statement" in prompt

    def test_contains_question(self):
        """Prompt should contain the question."""
        prompt = create_single_question_prompt(
            "Test statement", "Test question?", "binary"
        )
        assert "Test question?" in prompt

    def test_contains_scale_instruction(self):
        """Prompt should contain scale instruction."""
        prompt = create_single_question_prompt(
            "Test statement", "Test question?", "binary"
        )
        assert "0" in prompt
        assert "1" in prompt

    def test_asks_for_json(self):
        """Prompt should request JSON response."""
        prompt = create_single_question_prompt(
            "Test statement", "Test question?", "binary"
        )
        assert "JSON" in prompt or "json" in prompt


class TestBatchedPrompt:
    def test_contains_all_questions(self):
        """Batched prompt should contain all questions."""
        questions = [
            {"question": "Q1?"},
            {"question": "Q2?"},
            {"question": "Q3?"},
        ]
        prompt = create_batched_prompt("Statement", questions, "binary")

        assert "Q1?" in prompt
        assert "Q2?" in prompt
        assert "Q3?" in prompt

    def test_numbered_questions(self):
        """Questions should be numbered."""
        questions = [{"question": f"Question {i}?"} for i in range(3)]
        prompt = create_batched_prompt("Statement", questions, "binary")

        assert "1." in prompt
        assert "2." in prompt
        assert "3." in prompt

    def test_asks_for_scores_array(self):
        """Prompt should request scores array."""
        questions = [{"question": "Q?"}]
        prompt = create_batched_prompt("Statement", questions, "binary")

        assert "scores" in prompt.lower()

    def test_respects_batch_size(self):
        """Should only include up to batch_size questions."""
        questions = [{"question": f"Q{i}?"} for i in range(30)]
        prompt = create_batched_prompt("Statement", questions, "binary", batch_size=20)

        assert "Q0?" in prompt
        assert "Q19?" in prompt
        assert "Q20?" not in prompt


class TestBinaryEvaluationPrompt:
    def test_simple_format(self):
        """Binary prompt should be simple."""
        prompt = create_binary_evaluation_prompt("Statement", "Question?")

        assert "Statement" in prompt
        assert "Question?" in prompt
        assert "0" in prompt
        assert "1" in prompt


class TestExpectedFormat:
    def test_single_binary(self):
        """Single binary should show score options."""
        fmt = get_expected_response_format("binary", 1)
        assert "0" in fmt
        assert "1" in fmt

    def test_batch_format(self):
        """Batch should show scores array."""
        fmt = get_expected_response_format("binary", 20)
        assert "scores" in fmt.lower()
        assert "20" in fmt
