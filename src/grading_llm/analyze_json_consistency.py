"""
Analyze JSON consistency results: per-question score stability vs token confidence.

For each (text, scale, question):
- How many of 5 runs gave a different score?
- What was the token probability for that question's score?
- Does low probability predict score instability?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent.parent.parent / "data"

QUESTION_FAMILIES = [
    "named_entities", "actions_events", "causality", "temporal", "spatial",
    "numeric", "negation", "uncertainty", "modality", "sentiment",
    "emotion", "social", "dialogue", "first_person", "imperative",
    "comparison", "normative", "intent", "concreteness", "identity"
]
SCALE_ORDER = ["binary", "ternary", "quaternary", "continuous"]


def load_results():
    with open(DATA_DIR / "json_consistency_results.json") as f:
        return json.load(f)


def analyze():
    results = load_results()
    n_repeats = results["metadata"]["n_repeats"]
    n_questions = results["metadata"]["n_questions"]
    texts = list(results["per_text"].keys())

    # =========================================================================
    # 1. Per-question score differences (not byte-level, actual score values)
    # =========================================================================
    print("=" * 100)
    print("PART 1: Per-Question Score Stability (how many of 20 scores differ across 5 runs?)")
    print("=" * 100)

    # For each (text, scale): how many questions had at least 1 different score?
    print(f"\n{'Text':<25} {'Binary':>8} {'Ternary':>8} {'Quat':>8} {'Contin':>8} {'Total/80':>10}")
    print("-" * 100)

    total_diffs = {s: 0 for s in SCALE_ORDER}
    total_possible = len(texts) * n_questions

    for text_id in texts:
        row = f"{text_id:<25}"
        for scale in SCALE_ORDER:
            r = results["per_text"][text_id][scale]
            scores_by_run = r["parsed_scores"]  # list of 5 arrays, each length 20
            if len(scores_by_run) < 2:
                row += f"{'N/A':>8}"
                continue
            n_diff = 0
            for q_idx in range(n_questions):
                vals = [scores_by_run[rep][q_idx] for rep in range(len(scores_by_run))]
                if len(set(vals)) > 1:
                    n_diff += 1
            total_diffs[scale] += n_diff
            row += f"{n_diff:>6}/20"
        total_row = sum(
            sum(1 for q_idx in range(n_questions)
                if len(set(results["per_text"][text_id][s]["parsed_scores"][rep][q_idx]
                           for rep in range(len(results["per_text"][text_id][s]["parsed_scores"]))))
                > 1)
            for s in SCALE_ORDER
        )
        row += f"{total_row:>8}/80"
        print(row)

    print("-" * 100)
    row = f"{'TOTAL'::<25}"
    for scale in SCALE_ORDER:
        pct = total_diffs[scale] / total_possible * 100
        row += f"{total_diffs[scale]:>4}/{total_possible} "
    print(row)
    print()
    for scale in SCALE_ORDER:
        pct = total_diffs[scale] / total_possible * 100
        print(f"  {scale:<12}: {total_diffs[scale]}/{total_possible} questions unstable ({pct:.1f}%)")

    # =========================================================================
    # 2. Per-question confidence vs stability
    # =========================================================================
    print()
    print("=" * 100)
    print("PART 2: Token Probability vs Score Stability")
    print("  For each (text, scale, question): is the score stable? What was the token prob?")
    print("=" * 100)

    # Collect all (stable/unstable, token_prob) pairs
    stable_probs = []
    unstable_probs = []
    all_records = []  # (text, scale, q_idx, family, stable, mean_prob, score_variance, scores)

    for text_id in texts:
        for scale in SCALE_ORDER:
            r = results["per_text"][text_id][scale]
            scores_by_run = r["parsed_scores"]
            per_rep_conf = r.get("per_rep_confidence", [])

            for q_idx in range(n_questions):
                # Get scores across runs
                vals = [scores_by_run[rep][q_idx] for rep in range(len(scores_by_run))]
                is_stable = len(set(round(v, 4) for v in vals)) == 1
                score_var = np.var(vals) if vals else 0

                # Get token probability for this question across runs
                q_probs = []
                for rep_conf in per_rep_conf:
                    if rep_conf and "per_token" in rep_conf:
                        tokens = rep_conf["per_token"]
                        if q_idx < len(tokens):
                            q_probs.append(tokens[q_idx]["chosen_prob"])

                mean_prob = np.mean(q_probs) if q_probs else None
                min_prob = min(q_probs) if q_probs else None

                if mean_prob is not None:
                    if is_stable:
                        stable_probs.append(mean_prob)
                    else:
                        unstable_probs.append(mean_prob)

                all_records.append({
                    "text": text_id,
                    "scale": scale,
                    "q_idx": q_idx,
                    "family": QUESTION_FAMILIES[q_idx],
                    "stable": is_stable,
                    "mean_prob": mean_prob,
                    "min_prob": min_prob,
                    "score_var": round(float(score_var), 8),
                    "scores": [round(v, 4) for v in vals],
                })

    print(f"\nStable question-scores:   n={len(stable_probs)}, mean_prob={np.mean(stable_probs):.4f}, median={np.median(stable_probs):.4f}")
    print(f"Unstable question-scores: n={len(unstable_probs)}, mean_prob={np.mean(unstable_probs):.4f}, median={np.median(unstable_probs):.4f}")
    if stable_probs and unstable_probs:
        from scipy.stats import mannwhitneyu
        u_stat, p_val = mannwhitneyu(unstable_probs, stable_probs, alternative='less')
        print(f"\nMann-Whitney U test (H1: unstable tokens have LOWER prob than stable):")
        print(f"  U={u_stat:.1f}, p={p_val:.6f} {'***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'ns'}")

    # =========================================================================
    # 3. Per-question family breakdown
    # =========================================================================
    print()
    print("=" * 100)
    print("PART 3: Per-Question Reliability Profile")
    print("  Across all 10 texts and 4 scales (40 measurements per question)")
    print("=" * 100)

    q_stats = defaultdict(lambda: {"stable": 0, "unstable": 0, "probs": [], "unstable_probs": [], "vars": []})

    for rec in all_records:
        family = rec["family"]
        q_stats[family]["vars"].append(rec["score_var"])
        if rec["mean_prob"] is not None:
            q_stats[family]["probs"].append(rec["mean_prob"])
        if rec["stable"]:
            q_stats[family]["stable"] += 1
        else:
            q_stats[family]["unstable"] += 1
            if rec["mean_prob"] is not None:
                q_stats[family]["unstable_probs"].append(rec["mean_prob"])

    print(f"\n{'Question':<20} {'Stable':>7} {'Unstable':>9} {'% Unstable':>11} {'Mean Prob':>10} {'Mean Var':>10} {'Verdict':>12}")
    print("-" * 100)

    sorted_families = sorted(q_stats.keys(), key=lambda f: q_stats[f]["unstable"], reverse=True)
    for family in sorted_families:
        s = q_stats[family]
        total = s["stable"] + s["unstable"]
        pct_unstable = s["unstable"] / total * 100 if total > 0 else 0
        mean_prob = np.mean(s["probs"]) if s["probs"] else 0
        mean_var = np.mean(s["vars"]) if s["vars"] else 0
        verdict = "UNRELIABLE" if pct_unstable > 15 else "BORDERLINE" if pct_unstable > 5 else "RELIABLE"
        print(f"{family:<20} {s['stable']:>5}/40 {s['unstable']:>7}/40 {pct_unstable:>9.1f}% {mean_prob:>10.4f} {mean_var:>10.6f} {verdict:>12}")

    # =========================================================================
    # 4. Per-scale x per-question heatmap data
    # =========================================================================
    print()
    print("=" * 100)
    print("PART 4: Instability Rate by Question x Scale (% of 10 texts where score changed)")
    print("=" * 100)

    print(f"\n{'Question':<20} {'Binary':>8} {'Ternary':>8} {'Quat':>8} {'Contin':>8} {'Total':>8}")
    print("-" * 70)

    for q_idx, family in enumerate(QUESTION_FAMILIES):
        row = f"{family:<20}"
        total_unstable = 0
        for scale in SCALE_ORDER:
            n_unstable = 0
            for text_id in texts:
                rec = next(r for r in all_records
                           if r["text"] == text_id and r["scale"] == scale and r["q_idx"] == q_idx)
                if not rec["stable"]:
                    n_unstable += 1
            total_unstable += n_unstable
            pct = n_unstable / len(texts) * 100
            row += f"{n_unstable:>6}/10"
        row += f"{total_unstable:>6}/40"
        print(row)

    # =========================================================================
    # 5. Confidence breakdown for unstable entries
    # =========================================================================
    print()
    print("=" * 100)
    print("PART 5: Every Unstable (text, scale, question) with Confidence Detail")
    print("=" * 100)

    unstable_records = [r for r in all_records if not r["stable"]]
    unstable_records.sort(key=lambda r: r["score_var"], reverse=True)

    print(f"\n{'Text':<25} {'Scale':<12} {'Question':<20} {'Scores':<30} {'MeanProb':>9} {'MinProb':>8} {'Var':>10}")
    print("-" * 120)
    for rec in unstable_records:
        scores_str = str(rec["scores"])
        if len(scores_str) > 28:
            scores_str = scores_str[:28] + ".."
        mp = f"{rec['mean_prob']:.4f}" if rec['mean_prob'] else "N/A"
        mnp = f"{rec['min_prob']:.4f}" if rec['min_prob'] else "N/A"
        print(f"{rec['text']:<25} {rec['scale']:<12} {rec['family']:<20} {scores_str:<30} {mp:>9} {mnp:>8} {rec['score_var']:>10.6f}")

    # =========================================================================
    # 6. Probability threshold analysis
    # =========================================================================
    print()
    print("=" * 100)
    print("PART 6: If you threshold on token probability, what instability do you filter?")
    print("=" * 100)

    thresholds = [0.99, 0.98, 0.95, 0.90, 0.85, 0.80, 0.70, 0.50]
    print(f"\n{'Threshold':>10} {'Entries above':>14} {'Unstable above':>16} {'% Unstable':>12} {'Filtered out':>14}")
    print("-" * 80)

    all_with_prob = [(r["stable"], r["mean_prob"]) for r in all_records if r["mean_prob"] is not None]
    total_entries = len(all_with_prob)
    total_unstable_count = sum(1 for s, _ in all_with_prob if not s)

    for thresh in thresholds:
        above = [(s, p) for s, p in all_with_prob if p >= thresh]
        n_above = len(above)
        n_unstable_above = sum(1 for s, _ in above if not s)
        pct = n_unstable_above / n_above * 100 if n_above > 0 else 0
        filtered = total_entries - n_above
        print(f"{thresh:>10.2f} {n_above:>12}/{total_entries} {n_unstable_above:>14}/{n_above} {pct:>10.1f}% {filtered:>12} dropped")

    # Save analysis
    analysis_output = {
        "per_question_reliability": {
            family: {
                "stable_count": q_stats[family]["stable"],
                "unstable_count": q_stats[family]["unstable"],
                "pct_unstable": round(q_stats[family]["unstable"] / 40 * 100, 1),
                "mean_token_prob": round(float(np.mean(q_stats[family]["probs"])), 4) if q_stats[family]["probs"] else None,
                "mean_variance": round(float(np.mean(q_stats[family]["vars"])), 8),
            }
            for family in sorted_families
        },
        "confidence_vs_stability": {
            "stable_mean_prob": round(float(np.mean(stable_probs)), 4),
            "unstable_mean_prob": round(float(np.mean(unstable_probs)), 4),
            "mann_whitney_p": round(float(p_val), 8) if stable_probs and unstable_probs else None,
        },
        "unstable_entries": [
            {k: v for k, v in rec.items()}
            for rec in unstable_records
        ],
    }

    out_path = DATA_DIR / "json_consistency_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis_output, f, indent=2)
    print(f"\nSaved analysis to {out_path}")


if __name__ == "__main__":
    analyze()
