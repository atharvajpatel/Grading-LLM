"""
Report Generation

Generates comprehensive markdown reports with analysis results.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from .scales import SCALE_ORDER


def generate_auto_explanation(
    pca_metrics: Dict,
    n_samples_per_scale: int = 20,
    n_questions: int = 20
) -> str:
    """
    Generate the auto-explanation paragraph for the PCA plot.

    Args:
        pca_metrics: PCA metrics from analysis
        n_samples_per_scale: Number of samples per scale
        n_questions: Number of questions

    Returns:
        Explanation paragraph string
    """
    explained_var = pca_metrics.get("explained_variance_ratio", [0, 0, 0])
    top_loadings = pca_metrics.get("top_loading_questions", {})

    # Format top questions for each PC
    def format_top_questions(pc_key: str, n: int = 3) -> str:
        questions = top_loadings.get(pc_key, [])[:n]
        if not questions:
            return "N/A"
        return ", ".join([
            f'"{q["question"][:50]}..." ({q["loading"]:.2f})'
            for q in questions
        ])

    total_samples = n_samples_per_scale * len(SCALE_ORDER)

    return f"""This 3D PCA projection visualizes the geometric spread of {total_samples} embeddings
({n_samples_per_scale} per grading scale) across {n_questions} interpretable question dimensions for
the same statement. If the LLM were a perfectly consistent grader,
points from the same scale would collapse into a tight cluster.
Instead, dispersion patterns reveal where the model exhibits uncertainty.

**PC1** ({explained_var[0]*100:.1f}% variance) is primarily influenced by:
{format_top_questions("PC1")}

**PC2** ({explained_var[1]*100:.1f}% variance) is influenced by:
{format_top_questions("PC2")}

**PC3** ({explained_var[2]*100:.1f}% variance) is influenced by:
{format_top_questions("PC3")}

This analysis reveals where finer-grained grading exposes model uncertainty
and which semantic dimensions drive the observed instability."""


def generate_metrics_table(scale_metrics: Dict) -> str:
    """Generate markdown table of metrics by scale."""
    lines = [
        "| Scale | Avg Variance | Avg Consistency | Avg Entropy |",
        "|-------|--------------|-----------------|-------------|"
    ]

    for scale_name in SCALE_ORDER:
        if scale_name in scale_metrics:
            sm = scale_metrics[scale_name]
            lines.append(
                f"| {scale_name.capitalize()} | "
                f"{sm['avg_variance']:.4f} | "
                f"{sm['avg_consistency']:.2%} | "
                f"{sm['avg_entropy']:.3f} |"
            )

    return "\n".join(lines)


def generate_top_questions_section(pca_metrics: Dict) -> str:
    """Generate section for top contributing questions per PC."""
    lines = ["## Top Contributing Questions per Principal Component", ""]

    top_loadings = pca_metrics.get("top_loading_questions", {})
    explained_var = pca_metrics.get("explained_variance_ratio", [0, 0, 0])

    for i, pc_key in enumerate(["PC1", "PC2", "PC3"]):
        questions = top_loadings.get(pc_key, [])
        var_pct = explained_var[i] * 100 if i < len(explained_var) else 0

        lines.append(f"### {pc_key} ({var_pct:.1f}% variance)")
        lines.append("")

        for rank, q in enumerate(questions[:8], 1):
            loading_sign = "+" if q["loading"] > 0 else ""
            lines.append(
                f"{rank}. **{q['id']}**: {q['question']} "
                f"(loading: {loading_sign}{q['loading']:.3f})"
            )

        lines.append("")

    return "\n".join(lines)


def generate_report(
    statement: str,
    metrics: Dict,
    config: Dict,
    output_path: Path
) -> str:
    """
    Generate complete markdown report.

    Args:
        statement: The analyzed statement
        metrics: Complete metrics dictionary
        config: Run configuration
        output_path: Path to save report

    Returns:
        Report content string
    """
    scale_metrics = metrics.get("scale_metrics", {})
    pca_metrics = metrics.get("pca", {})
    aggregate = metrics.get("aggregate", {})

    # Generate sections
    explanation = generate_auto_explanation(
        pca_metrics,
        n_samples_per_scale=config.get("n_samples", 20),
        n_questions=aggregate.get("n_questions", 20)
    )

    metrics_table = generate_metrics_table(scale_metrics)
    top_questions = generate_top_questions_section(pca_metrics)

    # Build report
    report = f"""# QA-Embedding Stability Analysis

## Statement

> {statement}

## Configuration

- **Model**: {config.get("model", "gpt-4o-mini")}
- **Samples per scale**: {config.get("n_samples", 20)}
- **Number of questions**: {aggregate.get("n_questions", 20)}
- **Scales**: {", ".join(SCALE_ORDER)}
- **Timestamp**: {config.get("timestamp", datetime.now().isoformat())}

## 3D PCA Visualization

![PCA Plot](pca_plot.png)

### Interpretation

{explanation}

## Metrics Summary

{metrics_table}

{top_questions}

## Aggregate Statistics

- **Total samples analyzed**: {aggregate.get("total_samples", 0)}
- **Overall average variance**: {aggregate.get("overall_avg_variance", 0):.4f}
- **Overall average consistency**: {aggregate.get("overall_avg_consistency", 0):.2%}
- **Total explained variance (PC1-3)**: {sum(pca_metrics.get("explained_variance_ratio", [0, 0, 0]))*100:.1f}%

---

*Generated by GRADING-LLM*
"""

    # Save report
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    return report


def generate_batch_summary(
    statements: List[Dict],
    run_folder: Path,
    config: Dict
) -> str:
    """
    Generate summary report for batch processing.

    Args:
        statements: List of statement dicts with results
        run_folder: Path to run folder
        config: Run configuration

    Returns:
        Summary content string
    """
    lines = [
        "# Batch Analysis Summary",
        "",
        f"## Configuration",
        f"- **Model**: {config.get('model', 'gpt-4o-mini')}",
        f"- **Samples per scale**: {config.get('n_samples', 20)}",
        f"- **Statements processed**: {len(statements)}",
        f"- **Timestamp**: {config.get('timestamp', datetime.now().isoformat())}",
        "",
        "## Statement Results",
        "",
        "| ID | Statement (truncated) | Avg Variance | Status |",
        "|----|-----------------------|--------------|--------|"
    ]

    for stmt in statements:
        stmt_id = stmt.get("id", "unknown")
        text = stmt.get("text", "")[:50] + ("..." if len(stmt.get("text", "")) > 50 else "")
        avg_var = stmt.get("avg_variance", 0)
        status = stmt.get("status", "completed")

        lines.append(f"| {stmt_id} | {text} | {avg_var:.4f} | {status} |")

    lines.extend([
        "",
        "## Individual Reports",
        "",
        "See subfolders for detailed analysis of each statement.",
        "",
        "---",
        "*Generated by GRADING-LLM*"
    ])

    summary = "\n".join(lines)

    # Save
    with open(run_folder / "batch_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)

    return summary
