"""
Analysis Module

Computes stability metrics and performs PCA analysis on QA embeddings.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from sklearn.decomposition import PCA

from .scales import SCALE_ORDER, is_discrete, get_scale_values


@dataclass
class QuestionMetrics:
    """Metrics for a single question across samples."""
    question_id: str
    mean: float
    variance: float
    std: float
    mode: Optional[float]
    mode_consistency: float  # % matching mode
    entropy: float
    invalid_rate: float


@dataclass
class ScaleMetrics:
    """Aggregate metrics for a scale."""
    scale_name: str
    avg_variance: float
    avg_consistency: float
    avg_entropy: float
    question_metrics: List[QuestionMetrics]


@dataclass
class PCAResult:
    """Results from PCA analysis."""
    coords: np.ndarray  # (n_samples, 3)
    loadings: np.ndarray  # (3, n_questions)
    explained_variance_ratio: np.ndarray  # (3,)
    scale_labels: List[str]  # Label for each sample


def compute_entropy(values: np.ndarray, bins: Optional[List[float]] = None) -> float:
    """
    Compute Shannon entropy of values.

    Args:
        values: Array of values
        bins: Optional bin edges for discretization

    Returns:
        Entropy value
    """
    if bins is not None:
        # Discrete case
        counts = np.zeros(len(bins))
        for v in values:
            # Find closest bin
            idx = np.argmin(np.abs(np.array(bins) - v))
            counts[idx] += 1
    else:
        # Continuous case - use histogram
        counts, _ = np.histogram(values, bins=10)

    # Normalize to probabilities
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros

    return -np.sum(probs * np.log2(probs))


def compute_question_metrics(
    responses: np.ndarray,
    question_id: str,
    scale_name: str,
    invalid_mask: Optional[np.ndarray] = None
) -> QuestionMetrics:
    """
    Compute metrics for a single question.

    Args:
        responses: Array of responses for this question
        question_id: ID of the question
        scale_name: Name of the scale used
        invalid_mask: Boolean mask of invalid responses

    Returns:
        QuestionMetrics object
    """
    # Handle invalid responses
    if invalid_mask is not None:
        valid_responses = responses[~invalid_mask]
        invalid_rate = invalid_mask.sum() / len(responses)
    else:
        valid_responses = responses
        invalid_rate = 0.0

    if len(valid_responses) == 0:
        return QuestionMetrics(
            question_id=question_id,
            mean=0.0,
            variance=0.0,
            std=0.0,
            mode=None,
            mode_consistency=0.0,
            entropy=0.0,
            invalid_rate=1.0
        )

    mean = float(np.mean(valid_responses))
    variance = float(np.var(valid_responses))
    std = float(np.std(valid_responses))

    # Mode and consistency
    if is_discrete(scale_name):
        scale_values = get_scale_values(scale_name)
        # Round to nearest scale value
        rounded = np.array([
            min(scale_values, key=lambda x: abs(x - v))
            for v in valid_responses
        ])
        mode_result = stats.mode(rounded, keepdims=True)
        mode = float(mode_result.mode[0])
        mode_consistency = float(mode_result.count[0] / len(rounded))
        entropy = compute_entropy(valid_responses, bins=scale_values)
    else:
        # Continuous: round to 1 decimal place for mode calculation
        rounded = np.round(valid_responses, decimals=1)
        mode_result = stats.mode(rounded, keepdims=True)
        mode = float(mode_result.mode[0])
        mode_consistency = float(mode_result.count[0] / len(rounded))
        entropy = compute_entropy(valid_responses)

    return QuestionMetrics(
        question_id=question_id,
        mean=mean,
        variance=variance,
        std=std,
        mode=mode,
        mode_consistency=mode_consistency,
        entropy=entropy,
        invalid_rate=invalid_rate
    )


def compute_scale_metrics(
    embeddings: np.ndarray,
    question_ids: List[str],
    scale_name: str
) -> ScaleMetrics:
    """
    Compute aggregate metrics for a scale.

    Args:
        embeddings: Array of shape (n_samples, n_questions)
        question_ids: List of question IDs
        scale_name: Name of the scale

    Returns:
        ScaleMetrics object
    """
    n_questions = embeddings.shape[1]
    question_metrics = []

    for i in range(n_questions):
        qm = compute_question_metrics(
            embeddings[:, i],
            question_ids[i],
            scale_name
        )
        question_metrics.append(qm)

    avg_variance = np.mean([qm.variance for qm in question_metrics])
    avg_consistency = np.mean([qm.mode_consistency for qm in question_metrics])
    avg_entropy = np.mean([qm.entropy for qm in question_metrics])

    return ScaleMetrics(
        scale_name=scale_name,
        avg_variance=avg_variance,
        avg_consistency=avg_consistency,
        avg_entropy=avg_entropy,
        question_metrics=question_metrics
    )


def run_pca(
    embeddings: Dict[str, np.ndarray],
    n_components: int = 3
) -> PCAResult:
    """
    Run PCA across all scales.

    Args:
        embeddings: Dict mapping scale_name -> array of shape (n_samples, n_questions)
        n_components: Number of PCA components

    Returns:
        PCAResult object
    """
    # Stack all embeddings
    all_data = []
    scale_labels = []

    for scale_name in SCALE_ORDER:
        if scale_name in embeddings:
            data = embeddings[scale_name]
            all_data.append(data)
            scale_labels.extend([scale_name] * data.shape[0])

    X = np.vstack(all_data)

    # Run PCA
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(X)

    return PCAResult(
        coords=coords,
        loadings=pca.components_,
        explained_variance_ratio=pca.explained_variance_ratio_,
        scale_labels=scale_labels
    )


def get_top_loading_questions(
    loadings: np.ndarray,
    question_ids: List[str],
    questions: List[Dict],
    n_top: int = 8
) -> Dict[int, List[Dict]]:
    """
    Get top contributing questions for each PC.

    Args:
        loadings: Array of shape (n_components, n_questions)
        question_ids: List of question IDs
        questions: List of question dictionaries
        n_top: Number of top questions to return

    Returns:
        Dict mapping PC index (0-based) to list of question info dicts
    """
    result = {}

    for pc_idx in range(loadings.shape[0]):
        pc_loadings = loadings[pc_idx]

        # Get indices sorted by absolute loading
        sorted_indices = np.argsort(np.abs(pc_loadings))[::-1]

        top_questions = []
        for idx in sorted_indices[:n_top]:
            qid = question_ids[idx]
            loading = float(pc_loadings[idx])

            # Find question text
            q_dict = next(
                (q for q in questions if q.get("id") == qid),
                {"question": qid}
            )

            top_questions.append({
                "id": qid,
                "question": q_dict.get("question", ""),
                "family": q_dict.get("family", ""),
                "loading": loading,
                "abs_loading": abs(loading)
            })

        result[pc_idx] = top_questions

    return result


def count_identical_vectors(embeddings: Dict[str, np.ndarray]) -> Dict:
    """
    Count the number of DISTINCT vector patterns per scale.

    Examples (for 20 samples):
    - All 20 identical → 1 unique (min possible)
    - 10 of type A, 10 of type B → 2 unique
    - 18 of type A, 1 of type B, 1 of type C → 3 unique
    - All 20 different → 20 unique (max possible)

    Args:
        embeddings: Dict mapping scale_name -> array of shape (n_samples, n_questions)

    Returns:
        Dict with counts per scale and overall totals
    """
    result = {"per_scale": {}, "total_unique": 0, "total_samples": 0}
    all_vectors = []

    for scale_name, data in embeddings.items():
        n_samples = data.shape[0]

        # Round to avoid floating point comparison issues
        rounded = np.round(data, decimals=4)

        # Convert to tuples for hashing and count distinct patterns
        vector_tuples = [tuple(row) for row in rounded]
        distinct_patterns = set(vector_tuples)
        n_unique = len(distinct_patterns)

        result["per_scale"][scale_name] = {
            "total_samples": n_samples,
            "unique_vectors": n_unique,
            "identical_count": n_samples - n_unique,
        }

        all_vectors.extend(vector_tuples)

    # Global count across all scales - count distinct patterns
    result["total_samples"] = len(all_vectors)
    result["total_unique"] = len(set(all_vectors))

    return result


def compute_all_metrics(
    embeddings: Dict[str, np.ndarray],
    questions: List[Dict]
) -> Dict:
    """
    Compute all metrics for a full experiment.

    Args:
        embeddings: Dict mapping scale_name -> array
        questions: List of question dictionaries

    Returns:
        Complete metrics dictionary
    """
    question_ids = [q.get("id", f"q_{i}") for i, q in enumerate(questions)]

    # Per-scale metrics
    scale_metrics = {}
    for scale_name, data in embeddings.items():
        sm = compute_scale_metrics(data, question_ids, scale_name)
        scale_metrics[scale_name] = {
            "avg_variance": sm.avg_variance,
            "avg_consistency": sm.avg_consistency,
            "avg_entropy": sm.avg_entropy,
            "question_metrics": [
                {
                    "question_id": qm.question_id,
                    "mean": qm.mean,
                    "variance": qm.variance,
                    "std": qm.std,
                    "mode": qm.mode,
                    "mode_consistency": qm.mode_consistency,
                    "entropy": qm.entropy,
                    "invalid_rate": qm.invalid_rate
                }
                for qm in sm.question_metrics
            ]
        }

    # PCA analysis
    pca_result = run_pca(embeddings)
    top_loadings = get_top_loading_questions(
        pca_result.loadings, question_ids, questions
    )

    pca_metrics = {
        "explained_variance_ratio": pca_result.explained_variance_ratio.tolist(),
        "coords": pca_result.coords.tolist(),
        "scale_labels": pca_result.scale_labels,
        "top_loading_questions": {
            f"PC{i+1}": [
                {
                    "id": q["id"],
                    "question": q["question"],
                    "loading": q["loading"]
                }
                for q in questions_list
            ]
            for i, questions_list in top_loadings.items()
        }
    }

    # Aggregate metrics
    aggregate = {
        "total_samples": sum(
            embeddings[s].shape[0] for s in embeddings
        ),
        "n_questions": len(questions),
        "n_scales": len(embeddings),
        "overall_avg_variance": np.mean([
            sm["avg_variance"] for sm in scale_metrics.values()
        ]),
        "overall_avg_consistency": np.mean([
            sm["avg_consistency"] for sm in scale_metrics.values()
        ]),
    }

    # Count identical vectors
    identical_counts = count_identical_vectors(embeddings)

    # Include raw embeddings for heatmap visualization
    embeddings_raw = {
        scale_name: data.tolist()
        for scale_name, data in embeddings.items()
    }

    return {
        "scale_metrics": scale_metrics,
        "pca": pca_metrics,
        "aggregate": aggregate,
        "identical_counts": identical_counts,
        "embeddings_raw": embeddings_raw,
    }
