"""Tests for analysis module."""

import numpy as np
import pytest

from grading_llm.analysis import (
    compute_entropy,
    compute_question_metrics,
    compute_scale_metrics,
    run_pca,
    get_top_loading_questions,
)


class TestEntropy:
    def test_constant_low_entropy(self):
        """Constant values should have zero entropy."""
        values = np.array([1, 1, 1, 1, 1])
        entropy = compute_entropy(values, bins=[0, 1])
        assert entropy == 0.0

    def test_uniform_high_entropy(self):
        """Uniform distribution should have maximum entropy."""
        values = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        entropy = compute_entropy(values, bins=[0, 1])
        assert entropy > 0.9  # Should be close to 1.0 for binary


class TestQuestionMetrics:
    def test_constant_responses(self):
        """Constant responses should have zero variance."""
        responses = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        metrics = compute_question_metrics(responses, "q1", "binary")

        assert metrics.mean == 1.0
        assert metrics.variance == 0.0
        assert metrics.std == 0.0

    def test_variable_responses(self):
        """Variable responses should have non-zero variance."""
        responses = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
        metrics = compute_question_metrics(responses, "q1", "binary")

        assert metrics.mean == 0.6
        assert metrics.variance > 0
        assert metrics.mode == 1.0
        assert metrics.mode_consistency == 0.6

    def test_invalid_rate(self):
        """Invalid mask should affect invalid_rate."""
        responses = np.array([0.0, 1.0, 0.5, 1.0, 0.0])
        invalid_mask = np.array([False, False, True, False, False])
        metrics = compute_question_metrics(
            responses, "q1", "binary", invalid_mask
        )

        assert metrics.invalid_rate == 0.2


class TestScaleMetrics:
    def test_aggregate_metrics(self):
        """Scale metrics should aggregate question metrics."""
        embeddings = np.array([
            [0, 1, 0.5],
            [0, 1, 0.5],
            [1, 0, 0.5],
            [1, 0, 0.5],
        ])
        question_ids = ["q1", "q2", "q3"]

        metrics = compute_scale_metrics(embeddings, question_ids, "binary")

        assert metrics.scale_name == "binary"
        assert len(metrics.question_metrics) == 3
        assert metrics.avg_variance >= 0


class TestPCA:
    def test_pca_output_shape(self):
        """PCA should return correct shapes."""
        embeddings = {
            "binary": np.random.rand(10, 20),
            "ternary": np.random.rand(10, 20),
            "quaternary": np.random.rand(10, 20),
            "continuous": np.random.rand(10, 20),
        }

        result = run_pca(embeddings)

        assert result.coords.shape == (40, 3)  # 10 * 4 scales
        assert result.loadings.shape == (3, 20)  # 3 PCs, 20 questions
        assert len(result.explained_variance_ratio) == 3
        assert len(result.scale_labels) == 40

    def test_pca_labels(self):
        """PCA should assign correct scale labels."""
        embeddings = {
            "binary": np.random.rand(5, 10),
            "ternary": np.random.rand(5, 10),
        }

        result = run_pca(embeddings)

        assert result.scale_labels[:5] == ["binary"] * 5
        assert result.scale_labels[5:] == ["ternary"] * 5


class TestTopLoadings:
    def test_top_loadings_format(self):
        """Top loadings should return correct format."""
        loadings = np.array([
            [0.5, -0.3, 0.8, -0.1],
            [0.2, 0.7, -0.4, 0.1],
            [-0.1, 0.1, 0.2, 0.9],
        ])
        question_ids = ["q1", "q2", "q3", "q4"]
        questions = [
            {"id": "q1", "question": "Question 1", "family": "f1"},
            {"id": "q2", "question": "Question 2", "family": "f2"},
            {"id": "q3", "question": "Question 3", "family": "f3"},
            {"id": "q4", "question": "Question 4", "family": "f4"},
        ]

        result = get_top_loading_questions(loadings, question_ids, questions, n_top=2)

        assert 0 in result
        assert 1 in result
        assert 2 in result
        assert len(result[0]) == 2

        # PC0 top loading should be q3 (0.8)
        assert result[0][0]["id"] == "q3"
