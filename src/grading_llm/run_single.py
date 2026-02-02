"""
Single Statement CLI

Process one statement through the full GRADING-LLM analysis pipeline.

Usage:
    python -m grading_llm.run_single --text "Your statement here"
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .analysis import compute_all_metrics, run_pca
from .client import QAEmbeddingClient
from .config import get_api_key
from .io import (
    create_run_folder,
    save_config,
    save_embeddings,
    save_metrics,
    append_response,
)
from .pca_viz import create_pca_plot, create_variance_plot
from .questions import load_questions
from .report import generate_report
from .scales import SCALE_ORDER


async def run_single_statement(
    statement: str,
    n_samples: int = 20,
    model: str = "gpt-4o-mini",
) -> Path:
    """
    Run the full analysis pipeline for a single statement.

    Args:
        statement: Statement to analyze
        n_samples: Number of samples per scale
        model: Model to use

    Returns:
        Path to the run folder
    """
    # Validate API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load questions
    questions = load_questions()
    n_questions = len(questions)
    print(f"Loaded {n_questions} questions")

    # Create run folder
    run_folder = create_run_folder("single")
    print(f"Output folder: {run_folder}")

    # Save config
    config = {
        "type": "single",
        "statement": statement,
        "model": model,
        "n_samples": n_samples,
        "n_questions": n_questions,
        "scales": SCALE_ORDER,
        "timestamp": datetime.now().isoformat(),
    }
    save_config(config, run_folder)

    # Initialize client
    client = QAEmbeddingClient(
        api_key=api_key,
        model=model,
        batch_size=20,
    )

    # Response logging
    responses_path = run_folder / "responses.jsonl"

    def log_response(response):
        append_response(response, responses_path)

    # Generate embeddings for all scales
    print(f"\nGenerating embeddings ({n_samples} samples × {len(SCALE_ORDER)} scales × {n_questions} questions)...")
    print(f"Total API calls: {n_samples * len(SCALE_ORDER) * (n_questions // 20 + (1 if n_questions % 20 else 0))}")

    embeddings = {}
    for scale_name in SCALE_ORDER:
        print(f"\n  Processing {scale_name} scale...")
        embeddings[scale_name] = await client.query_all_questions(
            statement,
            questions,
            scale_name,
            n_samples=n_samples,
            on_response=log_response,
        )
        print(f"    Completed {n_samples} samples")

    # Save embeddings
    embeddings_data = {
        "statement": statement,
        "scales": {
            scale_name: {
                "samples": data.tolist(),
                "mean": data.mean(axis=0).tolist(),
                "std": data.std(axis=0).tolist(),
            }
            for scale_name, data in embeddings.items()
        }
    }
    save_embeddings(embeddings_data, run_folder)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_all_metrics(embeddings, questions)
    save_metrics(metrics, run_folder)

    # Create PCA visualization
    print("Creating PCA visualization...")
    pca_result = run_pca(embeddings)
    create_pca_plot(
        coords=pca_result.coords,
        scale_labels=pca_result.scale_labels,
        explained_variance=pca_result.explained_variance_ratio,
        output_path=run_folder / "pca_plot.png",
    )

    # Create variance plot
    create_variance_plot(metrics, run_folder / "variance_plot.png")

    # Generate report
    print("Generating report...")
    generate_report(
        statement=statement,
        metrics=metrics,
        config=config,
        output_path=run_folder / "report.md",
    )

    print(f"\n✓ Analysis complete!")
    print(f"  Results saved to: {run_folder}")
    print(f"  - pca_plot.png: 3D visualization")
    print(f"  - report.md: Full analysis report")
    print(f"  - metrics.json: Raw metrics data")

    return run_folder


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze QA-embedding stability for a single statement"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Statement to analyze"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=20,
        help="Number of samples per scale (default: 20)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (default: gpt-4o-mini)"
    )

    args = parser.parse_args()

    # Run analysis
    asyncio.run(run_single_statement(
        statement=args.text,
        n_samples=args.samples,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
