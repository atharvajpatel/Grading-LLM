"""
Extract Research-Page-Equivalent Metrics from batch_results_with_logprobs.json
==============================================================================
Produces the same data structures used by ResearchTab.tsx so they can be
hardcoded into LogprobsTab.tsx.

Also runs the Tier 1-3 advanced stats on this data.

Output: prints JSON blobs ready to paste into TSX.
"""

import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

SCALE_ORDER = ["binary", "ternary", "quaternary", "continuous"]
QUESTION_FAMILIES = [
    "named_entities", "actions_events", "causality", "temporal", "spatial",
    "numeric", "negation", "uncertainty", "modality", "sentiment",
    "emotion", "social", "dialogue", "first_person", "imperative",
    "comparison", "normative", "intent", "concreteness", "identity"
]
QUESTION_LABELS = [
    "Named Entities", "Actions/Events", "Causality", "Temporal", "Spatial",
    "Numeric", "Negation", "Uncertainty", "Modality", "Sentiment",
    "Emotion", "Social", "Dialogue", "First Person", "Imperative",
    "Comparison", "Normative", "Intent", "Concreteness", "Identity"
]


def load_logprobs_data(path):
    with open(path) as f:
        raw = json.load(f)
    texts = raw["texts"]
    scores = {}
    for item in texts:
        tid = item["id"]
        scores[tid] = {}
        for scale in SCALE_ORDER:
            scores[tid][scale] = np.array(item["raw_scores"][scale])
    text_ids = [item["id"] for item in texts]
    return scores, text_ids, raw


def compute_heatmap_data(scores, text_ids):
    """Mean variance per question across all texts, by scale."""
    rows = []
    for q_idx, label in enumerate(QUESTION_LABELS):
        row = {"question": label}
        for scale in SCALE_ORDER:
            variances = []
            for tid in text_ids:
                data = scores[tid][scale][:, q_idx]
                variances.append(float(np.var(data, ddof=0)))
            row[scale] = round(float(np.mean(variances)), 6)
        rows.append(row)
    return rows


def compute_scale_summary(scores, text_ids):
    """Aggregate metrics per scale."""
    rows = []
    for scale in SCALE_ORDER:
        all_variances = []
        all_consistencies = []
        for tid in text_ids:
            data = scores[tid][scale]
            for q_idx in range(20):
                col = data[:, q_idx]
                v = float(np.var(col, ddof=0))
                all_variances.append(v)
                vals, counts = np.unique(col, return_counts=True)
                all_consistencies.append(float(np.max(counts) / len(col)))

        variances_arr = np.array(all_variances)
        rows.append({
            "scale": scale,
            "meanVar": round(float(np.mean(variances_arr)), 6),
            "medianVar": round(float(np.median(variances_arr)), 6),
            "zeroVarPct": round(float(np.mean(variances_arr == 0)) * 100, 1),
            "highVarPct": round(float(np.mean(variances_arr > 0.01)) * 100, 1),
            "medianEntropy": 0.0,  # would need scipy - skip for display
            "meanConsistency": round(float(np.mean(all_consistencies)), 4),
        })
    return rows


def compute_degradation_data(scores, text_ids):
    """Per-text mean variance by scale."""
    rows = []
    for scale in SCALE_ORDER:
        row = {"scale": scale}
        for tid in text_ids:
            data = scores[tid][scale]
            variances = [float(np.var(data[:, q], ddof=0)) for q in range(20)]
            row[tid] = round(float(np.mean(variances)), 6)
        rows.append(row)
    return rows


def compute_text_difficulty(scores, text_ids):
    """Texts ranked by overall mean variance."""
    results = []
    for tid in text_ids:
        all_var = []
        for scale in SCALE_ORDER:
            data = scores[tid][scale]
            for q in range(20):
                all_var.append(float(np.var(data[:, q], ddof=0)))
        results.append({
            "id": tid,
            "meanVariance": round(float(np.mean(all_var)), 6),
        })
    results.sort(key=lambda x: x["meanVariance"], reverse=True)
    return results


def compute_mode_consistency_data(scores, text_ids):
    """Mean mode consistency per text per scale."""
    rows = []
    for tid in text_ids:
        row = {"id": tid}
        for scale in SCALE_ORDER:
            data = scores[tid][scale]
            consistencies = []
            for q in range(20):
                col = data[:, q]
                vals, counts = np.unique(col, return_counts=True)
                consistencies.append(float(np.max(counts) / len(col)))
            row[scale] = round(float(np.mean(consistencies)), 4)
        rows.append(row)
    return rows


def compute_per_text_table(scores, text_ids):
    """Per-text, per-scale metrics table."""
    rows = []
    for tid in text_ids:
        for scale in SCALE_ORDER:
            data = scores[tid][scale]
            variances = []
            consistencies = []
            entropies = []
            for q in range(20):
                col = data[:, q]
                variances.append(float(np.var(col, ddof=0)))
                vals, counts = np.unique(col, return_counts=True)
                consistencies.append(float(np.max(counts) / len(col)))
                # simple entropy
                probs = counts / counts.sum()
                ent = -float(np.sum(probs * np.log2(probs + 1e-12)))
                entropies.append(ent)

            # unique vectors
            rounded = np.round(data, 4)
            unique = len(set(tuple(r) for r in rounded))

            rows.append({
                "textId": tid,
                "scale": scale,
                "avgVariance": round(float(np.mean(variances)), 6),
                "avgConsistency": round(float(np.mean(consistencies)), 4),
                "avgEntropy": round(float(np.mean(entropies)), 4),
                "uniqueVectors": unique,
            })
    return rows


def run_advanced_stats_on_logprobs(scores, text_ids):
    """Run Tier 1-3 from advanced_stats.py on this data."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from advanced_stats import compute_tier1, compute_tier2, compute_tier3

    info = {
        "scores": scores,
        "text_ids": text_ids,
        "scale_order": SCALE_ORDER,
        "question_families": QUESTION_FAMILIES,
        "n_samples": 20,
        "n_questions": 20,
        "n_texts": len(text_ids),
    }

    tier1 = compute_tier1(info)
    tier2 = compute_tier2(info)
    tier3 = compute_tier3(info)
    return tier1, tier2, tier3


def fmt(obj):
    """Format JSON for TSX embedding."""
    return json.dumps(obj, indent=2)


def main():
    path = PROJECT_ROOT / "data" / "batch_results_with_logprobs.json"
    scores, text_ids, raw = load_logprobs_data(str(path))

    print("Computing research-page metrics from logprobs run...")

    heatmap = compute_heatmap_data(scores, text_ids)
    scale_summary = compute_scale_summary(scores, text_ids)
    degradation = compute_degradation_data(scores, text_ids)
    text_difficulty = compute_text_difficulty(scores, text_ids)
    mode_consistency = compute_mode_consistency_data(scores, text_ids)
    per_text_table = compute_per_text_table(scores, text_ids)

    print("Running advanced stats (Tier 1-3)...")
    tier1, tier2, tier3 = run_advanced_stats_on_logprobs(scores, text_ids)

    # Logprob summary from the raw file
    logprob_summary = raw.get("logprob_summary", {})

    # Build output
    output = {
        "metadata": raw["metadata"],
        "usage": raw["usage"],
        "heatmap_data": heatmap,
        "scale_summary": scale_summary,
        "degradation_data": degradation,
        "text_difficulty": text_difficulty,
        "mode_consistency_data": mode_consistency,
        "per_text_table": per_text_table,
        "logprob_summary": logprob_summary,
        "tier1": tier1,
        "tier2": tier2,
        "tier3": tier3,
    }

    out_path = PROJECT_ROOT / "data" / "logprobs_research_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Saved to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.1f} KB")

    # Print key numbers for verification
    print("\n=== SCALE SUMMARY ===")
    for s in scale_summary:
        print(f"  {s['scale']:12s}: meanVar={s['meanVar']:.6f}  zeroVar%={s['zeroVarPct']}  consistency={s['meanConsistency']:.4f}")

    print("\n=== TIER 1 (means) ===")
    for metric in ["cronbachs_alpha", "icc", "krippendorff_alpha"]:
        print(f"  {metric}:")
        for scale in SCALE_ORDER:
            val = tier1[metric][scale]["mean"]
            print(f"    {scale:12s}: {val:.4f}" if val else f"    {scale:12s}: N/A")

    print("\n=== TIER 3 FRIEDMAN ===")
    fr = tier3["friedman_test"]
    print(f"  chi2={fr['chi_squared']:.2f}, p={fr['p_value']:.2e}, sig={fr['significant_at_005']}")

    print("\n=== TIER 3 ETA-SQUARED ===")
    for item in tier3["effect_sizes_eta_squared"]["ranking"]:
        print(f"  {item['factor']:12s}: {item['eta_squared']:.4f}")


if __name__ == "__main__":
    main()
