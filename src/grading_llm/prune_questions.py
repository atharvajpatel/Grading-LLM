"""
Question Pruning CLI

Evaluates candidate questions on a synthetic corpus and selects
a low-redundancy subset using correlation-based pruning.

Usage:
    python -m grading_llm.prune_questions --candidates data/questions_100.json --out data/questions_20.json
    python -m grading_llm.prune_questions --dry_run
"""

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import get_api_key
from .io import create_run_folder, get_project_root
from .questiongen import load_candidate_questions, get_minimal_pair, FACTOR_FAMILIES
from .synth_corpus import (
    generate_synthetic_corpus,
    generate_minimal_pair_corpus,
    save_corpus,
)


async def evaluate_question_on_statement(
    client,
    statement: str,
    question: str,
    semaphore: asyncio.Semaphore
) -> int:
    """
    Evaluate a single question on a statement (binary, temp=0).

    Returns: 0 or 1
    """
    from openai import AsyncOpenAI

    prompt = f'''You are evaluating the following statement:
"{statement}"

Question: {question}

Respond with ONLY a JSON object: {{"answer": 0}} or {{"answer": 1}}

Score must be exactly 0 (no/absent) or 1 (yes/present).'''

    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=50,
            )

            content = response.choices[0].message.content.strip()

            # Parse JSON response
            import re
            match = re.search(r'"answer"\s*:\s*(\d)', content)
            if match:
                return int(match.group(1))

            # Fallback: look for 0 or 1
            if "1" in content:
                return 1
            return 0

        except Exception as e:
            print(f"  Error evaluating: {e}")
            return 0


async def evaluate_question_bank(
    questions: List[Dict],
    corpus: List[Dict],
    api_key: str,
    max_concurrent: int = 20
) -> np.ndarray:
    """
    Evaluate all questions on all corpus statements.

    Args:
        questions: List of question dictionaries
        corpus: List of statement dictionaries
        api_key: OpenAI API key
        max_concurrent: Max concurrent API calls

    Returns:
        answer_matrix: shape (n_statements, n_questions) with 0/1 values
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    n_statements = len(corpus)
    n_questions = len(questions)

    print(f"Evaluating {n_questions} questions on {n_statements} statements...")
    print(f"Total API calls: {n_statements * n_questions}")

    # Create matrix
    answer_matrix = np.zeros((n_statements, n_questions), dtype=np.int8)

    # Process in batches by statement
    for i, stmt in enumerate(corpus):
        if (i + 1) % 10 == 0:
            print(f"  Processing statement {i + 1}/{n_statements}...")

        tasks = []
        for j, q in enumerate(questions):
            task = evaluate_question_on_statement(
                client, stmt["text"], q["question"], semaphore
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        for j, result in enumerate(results):
            answer_matrix[i, j] = result

    return answer_matrix


async def check_minimal_pairs(
    questions: List[Dict],
    api_key: str,
    max_concurrent: int = 20
) -> Dict[str, int]:
    """
    Check minimal pair accuracy for each question.

    Returns: dict mapping question ID to score (0, 1, or 2)
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(max_concurrent)

    scores = {}

    print(f"Checking minimal pairs for {len(questions)} questions...")

    for q in questions:
        qid = q["id"]
        mp = get_minimal_pair(q)

        if not mp:
            scores[qid] = 0
            continue

        score = 0

        # Check positive example (should return 1)
        if "positive" in mp:
            result = await evaluate_question_on_statement(
                client, mp["positive"], q["question"], semaphore
            )
            if result == 1:
                score += 1

        # Check negative example (should return 0)
        if "negative" in mp:
            result = await evaluate_question_on_statement(
                client, mp["negative"], q["question"], semaphore
            )
            if result == 0:
                score += 1

        scores[qid] = score

    return scores


def compute_correlations(answer_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute correlation and Jaccard similarity matrices.

    Args:
        answer_matrix: shape (n_statements, n_questions)

    Returns:
        corr_matrix: Pearson correlation matrix
        jaccard_matrix: Jaccard similarity matrix
    """
    n_questions = answer_matrix.shape[1]

    # Pearson correlation (phi coefficient for binary)
    # Handle constant columns
    with np.errstate(invalid='ignore', divide='ignore'):
        corr_matrix = np.corrcoef(answer_matrix.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    # Jaccard similarity
    jaccard_matrix = np.zeros((n_questions, n_questions))
    for i in range(n_questions):
        for j in range(n_questions):
            intersection = np.sum(answer_matrix[:, i] & answer_matrix[:, j])
            union = np.sum(answer_matrix[:, i] | answer_matrix[:, j])
            if union > 0:
                jaccard_matrix[i, j] = intersection / union
            else:
                jaccard_matrix[i, j] = 0.0

    return corr_matrix, jaccard_matrix


def select_independent_questions(
    answer_matrix: np.ndarray,
    questions: List[Dict],
    minimal_pair_scores: Dict[str, int],
    k: int = 20,
    corr_threshold: float = 0.85,
    max_per_family: int = 2
) -> List[int]:
    """
    Greedy selection of low-redundancy questions.

    Args:
        answer_matrix: shape (n_statements, n_questions)
        questions: List of question dictionaries
        minimal_pair_scores: Dict mapping question ID to score
        k: Number of questions to select
        corr_threshold: Maximum correlation allowed
        max_per_family: Maximum questions per family

    Returns:
        List of selected question indices
    """
    n_questions = len(questions)

    # Compute correlations
    corr_matrix, _ = compute_correlations(answer_matrix)

    # Compute variances
    variances = answer_matrix.var(axis=0)

    # Track selection
    selected = []
    family_counts = defaultdict(int)

    # Start with question having variance closest to 0.25 (max entropy for binary)
    valid_first = [
        i for i, q in enumerate(questions)
        if variances[i] > 0.01  # Not constant
    ]
    if valid_first:
        first = min(valid_first, key=lambda i: abs(variances[i] - 0.25))
    else:
        first = 0

    selected.append(first)
    family_counts[questions[first]["family"]] += 1

    print(f"Starting with Q{first}: {questions[first]['id']}")

    while len(selected) < k:
        best_candidate = None
        best_score = -float('inf')

        for i in range(n_questions):
            if i in selected:
                continue

            q = questions[i]
            family = q["family"]

            # Check family limit
            if family_counts[family] >= max_per_family:
                continue

            # Check variance (skip constants)
            if variances[i] < 0.01:
                continue

            # Compute max correlation with selected
            max_corr = max(abs(corr_matrix[i, s]) for s in selected)

            if max_corr >= corr_threshold:
                continue  # Too correlated

            # Compute novelty score
            novelty = 1 - max_corr

            # Add minimal pair accuracy as tie-breaker
            mp_score = minimal_pair_scores.get(q["id"], 0)
            score = novelty + 0.1 * mp_score

            if score > best_score:
                best_score = score
                best_candidate = i

        if best_candidate is None:
            print(f"  No more valid candidates. Selected {len(selected)} questions.")
            break

        selected.append(best_candidate)
        family_counts[questions[best_candidate]["family"]] += 1
        print(f"  Selected Q{len(selected)}: {questions[best_candidate]['id']} "
              f"(family: {questions[best_candidate]['family']}, novelty: {1 - max(abs(corr_matrix[best_candidate, s]) for s in selected[:-1]):.3f})")

    return selected


def generate_pruning_report(
    questions: List[Dict],
    selected_indices: List[int],
    answer_matrix: np.ndarray,
    minimal_pair_scores: Dict[str, int],
    output_dir: Path
) -> str:
    """Generate a markdown report explaining the pruning results."""
    corr_matrix, jaccard_matrix = compute_correlations(answer_matrix)
    variances = answer_matrix.var(axis=0)

    lines = [
        "# Question Pruning Report",
        "",
        f"## Summary",
        f"- Candidate questions: {len(questions)}",
        f"- Selected questions: {len(selected_indices)}",
        f"- Corpus size: {answer_matrix.shape[0]} statements",
        "",
        "## Selected Questions",
        "",
    ]

    for rank, idx in enumerate(selected_indices, 1):
        q = questions[idx]
        mp_score = minimal_pair_scores.get(q["id"], 0)
        lines.append(f"### {rank}. {q['id']}")
        lines.append(f"- **Family**: {q['family']}")
        lines.append(f"- **Question**: {q['question']}")
        lines.append(f"- **Variance**: {variances[idx]:.3f}")
        lines.append(f"- **Minimal pair accuracy**: {mp_score}/2")
        lines.append("")

    # Family distribution
    lines.append("## Family Distribution")
    lines.append("")
    family_counts = defaultdict(int)
    for idx in selected_indices:
        family_counts[questions[idx]["family"]] += 1

    for family in sorted(family_counts.keys()):
        lines.append(f"- {family}: {family_counts[family]}")
    lines.append("")

    # Correlation analysis
    selected_corr = corr_matrix[np.ix_(selected_indices, selected_indices)]
    off_diag = selected_corr[np.triu_indices(len(selected_indices), k=1)]

    lines.append("## Correlation Analysis (Selected Questions)")
    lines.append("")
    lines.append(f"- Max off-diagonal correlation: {np.max(np.abs(off_diag)):.3f}")
    lines.append(f"- Mean off-diagonal correlation: {np.mean(np.abs(off_diag)):.3f}")
    lines.append("")

    report = "\n".join(lines)

    # Save report
    with open(output_dir / "pruning_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    return report


async def run_pruning_pipeline(
    candidates_path: Path,
    output_path: Path,
    k: int = 20,
    corr_threshold: float = 0.85,
    max_per_family: int = 2,
    dry_run: bool = False
) -> None:
    """Run the full pruning pipeline."""
    # Load candidates
    print(f"Loading candidates from {candidates_path}...")
    questions = load_candidate_questions(candidates_path)
    print(f"Loaded {len(questions)} candidate questions")

    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create run folder
    run_folder = create_run_folder("pruning")
    pruning_dir = run_folder / "pruning"
    pruning_dir.mkdir(exist_ok=True)
    print(f"Output folder: {run_folder}")

    # Generate synthetic corpus
    print("\nGenerating synthetic corpus...")
    corpus = generate_synthetic_corpus(n_base=30)

    # Add minimal pair corpus
    mp_corpus = generate_minimal_pair_corpus(questions)
    corpus.extend(mp_corpus)

    print(f"Generated {len(corpus)} statements")
    save_corpus(corpus, pruning_dir / "synthetic_corpus.jsonl")

    # Check minimal pairs
    print("\nChecking minimal pairs...")
    mp_scores = await check_minimal_pairs(questions, api_key)

    # Report minimal pair accuracy
    good_mp = sum(1 for s in mp_scores.values() if s == 2)
    print(f"Questions with perfect minimal pair accuracy: {good_mp}/{len(questions)}")

    # Evaluate questions on corpus
    print("\nEvaluating questions on corpus...")
    answer_matrix = await evaluate_question_bank(questions, corpus, api_key)

    # Save answer matrix
    np.save(pruning_dir / "answer_matrix.npy", answer_matrix)

    # Save as CSV for debugging
    import csv
    with open(pruning_dir / "answer_matrix.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["statement_id"] + [q["id"] for q in questions]
        writer.writerow(header)
        for i, stmt in enumerate(corpus):
            row = [stmt["id"]] + list(answer_matrix[i])
            writer.writerow(row)

    # Select independent questions
    print(f"\nSelecting {k} independent questions...")
    selected_indices = select_independent_questions(
        answer_matrix, questions, mp_scores,
        k=k, corr_threshold=corr_threshold, max_per_family=max_per_family
    )

    # Generate report
    print("\nGenerating pruning report...")
    report = generate_pruning_report(
        questions, selected_indices, answer_matrix, mp_scores, pruning_dir
    )
    print(report)

    if dry_run:
        print("\n[DRY RUN] Not saving output file.")
        return

    # Build output question bank
    selected_questions = [questions[i] for i in selected_indices]
    output_data = {
        "version": "1.0.0",
        "description": f"Pruned question bank ({len(selected_questions)} questions)",
        "source": str(candidates_path),
        "questions": selected_questions
    }

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved {len(selected_questions)} questions to {output_path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prune candidate questions to a low-redundancy subset"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/questions_100.json"),
        help="Path to candidate questions JSON"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/questions_20.json"),
        help="Output path for pruned questions"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=20,
        help="Number of questions to select"
    )
    parser.add_argument(
        "--corr_threshold",
        type=float,
        default=0.85,
        help="Maximum correlation allowed between questions"
    )
    parser.add_argument(
        "--max_per_family",
        type=int,
        default=2,
        help="Maximum questions per family"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only compute diagnostics, don't save output"
    )

    args = parser.parse_args()

    # Run async pipeline
    asyncio.run(run_pruning_pipeline(
        candidates_path=args.candidates,
        output_path=args.out,
        k=args.k,
        corr_threshold=args.corr_threshold,
        max_per_family=args.max_per_family,
        dry_run=args.dry_run
    ))


if __name__ == "__main__":
    main()
