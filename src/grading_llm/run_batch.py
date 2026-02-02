"""
Batch Processing CLI

Process multiple statements from a JSONL file.

Usage:
    python -m grading_llm.run_batch --input statements.jsonl
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path

from .analysis import compute_all_metrics, run_pca
from .client import QAEmbeddingClient
from .config import get_api_key
from .io import (
    create_run_folder,
    load_batch_input,
    save_config,
    save_embeddings,
    save_metrics,
    append_response,
)
from .pca_viz import create_pca_plot, create_variance_plot
from .questions import load_questions
from .report import generate_report, generate_batch_summary
from .scales import SCALE_ORDER


async def process_statement(
    statement: str,
    stmt_id: str,
    client: QAEmbeddingClient,
    questions: list,
    output_folder: Path,
    n_samples: int = 20,
    config: dict = None,
) -> dict:
    """
    Process a single statement within a batch.

    Returns:
        Dict with results summary
    """
    stmt_folder = output_folder / stmt_id
    stmt_folder.mkdir(exist_ok=True)

    # Response logging
    responses_path = stmt_folder / "responses.jsonl"

    def log_response(response):
        append_response(response, responses_path)

    # Generate embeddings
    embeddings = {}
    for scale_name in SCALE_ORDER:
        embeddings[scale_name] = await client.query_all_questions(
            statement,
            questions,
            scale_name,
            n_samples=n_samples,
            on_response=log_response,
        )

    # Save embeddings
    embeddings_data = {
        "statement": statement,
        "id": stmt_id,
        "scales": {
            scale_name: {
                "samples": data.tolist(),
                "mean": data.mean(axis=0).tolist(),
                "std": data.std(axis=0).tolist(),
            }
            for scale_name, data in embeddings.items()
        }
    }
    save_embeddings(embeddings_data, stmt_folder)

    # Compute metrics
    metrics = compute_all_metrics(embeddings, questions)
    save_metrics(metrics, stmt_folder)

    # Create PCA visualization
    pca_result = run_pca(embeddings)
    create_pca_plot(
        coords=pca_result.coords,
        scale_labels=pca_result.scale_labels,
        explained_variance=pca_result.explained_variance_ratio,
        output_path=stmt_folder / "pca_plot.png",
    )

    # Create variance plot
    create_variance_plot(metrics, stmt_folder / "variance_plot.png")

    # Generate report
    stmt_config = {**(config or {}), "statement_id": stmt_id}
    generate_report(
        statement=statement,
        metrics=metrics,
        config=stmt_config,
        output_path=stmt_folder / "report.md",
    )

    return {
        "id": stmt_id,
        "text": statement,
        "avg_variance": metrics.get("aggregate", {}).get("overall_avg_variance", 0),
        "status": "completed",
    }


async def run_batch(
    input_path: Path,
    n_samples: int = 20,
    model: str = "gpt-4o-mini",
) -> Path:
    """
    Run the analysis pipeline for a batch of statements.

    Args:
        input_path: Path to input JSONL file
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

    # Load input
    print(f"Loading statements from {input_path}...")
    statements = load_batch_input(input_path)
    print(f"Loaded {len(statements)} statements")

    # Load questions
    questions = load_questions()
    n_questions = len(questions)
    print(f"Loaded {n_questions} questions")

    # Create run folder
    run_folder = create_run_folder("batch")
    print(f"Output folder: {run_folder}")

    # Save config
    config = {
        "type": "batch",
        "input_file": str(input_path),
        "n_statements": len(statements),
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

    # Process each statement
    results = []
    total_calls = len(statements) * n_samples * len(SCALE_ORDER) * (n_questions // 20 + (1 if n_questions % 20 else 0))
    print(f"\nProcessing {len(statements)} statements...")
    print(f"Estimated API calls: {total_calls}")

    for i, stmt in enumerate(statements):
        stmt_id = stmt.get("id", f"stmt_{i:03d}")
        stmt_text = stmt["text"]

        print(f"\n[{i+1}/{len(statements)}] Processing {stmt_id}...")
        print(f"  \"{stmt_text[:60]}{'...' if len(stmt_text) > 60 else ''}\"")

        try:
            result = await process_statement(
                statement=stmt_text,
                stmt_id=stmt_id,
                client=client,
                questions=questions,
                output_folder=run_folder,
                n_samples=n_samples,
                config=config,
            )
            results.append(result)
            print(f"  ✓ Completed (avg variance: {result['avg_variance']:.4f})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                "id": stmt_id,
                "text": stmt_text,
                "avg_variance": 0,
                "status": f"error: {str(e)}",
            })

    # Generate batch summary
    print("\nGenerating batch summary...")
    generate_batch_summary(results, run_folder, config)

    print(f"\n✓ Batch processing complete!")
    print(f"  Results saved to: {run_folder}")
    print(f"  - batch_summary.md: Overview of all statements")
    print(f"  - <statement_id>/: Individual analysis folders")

    return run_folder


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze QA-embedding stability for a batch of statements"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to input JSONL file"
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

    # Validate input file
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    # Run batch processing
    asyncio.run(run_batch(
        input_path=args.input,
        n_samples=args.samples,
        model=args.model,
    ))


if __name__ == "__main__":
    main()
