"""
Compute all comparison metrics between Run 1 and Run 2 + logprob analysis.
Outputs data ready to hardcode into ComparisonTab.tsx.
"""

import json
import math
from pathlib import Path
from itertools import product

import numpy as np
from scipy import stats as sp_stats

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


def load_run1():
    with open(PROJECT_ROOT / "data" / "batch_results.json") as f:
        raw = json.load(f)
    scores = {}
    for item in raw:
        scores[item["id"]] = {}
        for scale in SCALE_ORDER:
            scores[item["id"]][scale] = np.array(item["raw_scores"][scale])
    return scores, [item["id"] for item in raw]


def load_run2():
    with open(PROJECT_ROOT / "data" / "batch_results_with_logprobs.json") as f:
        raw = json.load(f)
    scores = {}
    logprobs = {}
    for item in raw["texts"]:
        scores[item["id"]] = {}
        logprobs[item["id"]] = {}
        for scale in SCALE_ORDER:
            scores[item["id"]][scale] = np.array(item["raw_scores"][scale])
            logprobs[item["id"]][scale] = item["logprobs"][scale]
    return scores, logprobs, [item["id"] for item in raw["texts"]]


# ─── Section 1: Run-to-Run Reproducibility ───────────────────────────────────

def compute_scale_deltas(s1, s2, text_ids):
    """Side-by-side delta table per scale."""
    rows = []
    for scale in SCALE_ORDER:
        v1, v2, c1, c2 = [], [], [], []
        for tid in text_ids:
            for q in range(20):
                col1 = s1[tid][scale][:, q]
                col2 = s2[tid][scale][:, q]
                v1.append(float(np.var(col1, ddof=0)))
                v2.append(float(np.var(col2, ddof=0)))
                _, cnt1 = np.unique(col1, return_counts=True)
                _, cnt2 = np.unique(col2, return_counts=True)
                c1.append(float(np.max(cnt1) / len(col1)))
                c2.append(float(np.max(cnt2) / len(col2)))
        mv1, mv2 = np.mean(v1), np.mean(v2)
        mc1, mc2 = np.mean(c1), np.mean(c2)
        zv1 = np.mean(np.array(v1) == 0) * 100
        zv2 = np.mean(np.array(v2) == 0) * 100
        rows.append({
            "scale": scale,
            "run1MeanVar": round(float(mv1), 6),
            "run2MeanVar": round(float(mv2), 6),
            "deltaPct": round(float((mv2 - mv1) / mv1 * 100), 1) if mv1 > 0 else 0,
            "run1Consistency": round(float(mc1), 4),
            "run2Consistency": round(float(mc2), 4),
            "run1ZeroVarPct": round(float(zv1), 1),
            "run2ZeroVarPct": round(float(zv2), 1),
        })
    return rows


def compute_heatmap_diff(s1, s2, text_ids):
    """Variance diff: Run2 - Run1 per question x scale."""
    rows = []
    for q_idx, label in enumerate(QUESTION_LABELS):
        row = {"question": label}
        for scale in SCALE_ORDER:
            v1_list, v2_list = [], []
            for tid in text_ids:
                v1_list.append(float(np.var(s1[tid][scale][:, q_idx], ddof=0)))
                v2_list.append(float(np.var(s2[tid][scale][:, q_idx], ddof=0)))
            row[f"{scale}_run1"] = round(float(np.mean(v1_list)), 6)
            row[f"{scale}_run2"] = round(float(np.mean(v2_list)), 6)
            row[scale] = round(float(np.mean(v2_list) - np.mean(v1_list)), 6)
        rows.append(row)
    return rows


def compute_variance_scatter(s1, s2, text_ids):
    """800 points: (run1_variance, run2_variance) per (text, question, scale)."""
    points = []
    for tid in text_ids:
        for scale in SCALE_ORDER:
            for q in range(20):
                v1 = float(np.var(s1[tid][scale][:, q], ddof=0))
                v2 = float(np.var(s2[tid][scale][:, q], ddof=0))
                points.append({
                    "text": tid, "scale": scale,
                    "question": QUESTION_FAMILIES[q],
                    "run1Var": round(v1, 6), "run2Var": round(v2, 6),
                })

    # Correlation
    r1 = [p["run1Var"] for p in points]
    r2 = [p["run2Var"] for p in points]
    if np.std(r1) > 0 and np.std(r2) > 0:
        pearson_r, pearson_p = sp_stats.pearsonr(r1, r2)
    else:
        pearson_r, pearson_p = 1.0, 0.0

    return {
        "points": points,
        "pearsonR": round(float(pearson_r), 4),
        "pearsonP": float(pearson_p),
        "n": len(points),
    }


# ─── Section 3: Logprob Analysis ─────────────────────────────────────────────

def compute_confidence_heatmap(logprobs, text_ids):
    """Mean logprob confidence per question x scale."""
    rows = []
    for q_idx, label in enumerate(QUESTION_LABELS):
        row = {"question": label}
        for scale in SCALE_ORDER:
            probs = []
            for tid in text_ids:
                for sample_lps in logprobs[tid][scale]:
                    if q_idx < len(sample_lps):
                        probs.append(sample_lps[q_idx].get("probability", 1.0))
            row[scale] = round(float(np.mean(probs)), 4) if probs else 1.0
        rows.append(row)
    return rows


def compute_confidence_vs_variance(s2, logprobs, text_ids):
    """800 points: (mean_confidence, variance) per (text, question, scale)."""
    points = []
    for tid in text_ids:
        for scale in SCALE_ORDER:
            for q in range(20):
                v = float(np.var(s2[tid][scale][:, q], ddof=0))
                probs = []
                for sample_lps in logprobs[tid][scale]:
                    if q < len(sample_lps):
                        probs.append(sample_lps[q].get("probability", 1.0))
                mean_conf = float(np.mean(probs)) if probs else 1.0
                points.append({
                    "text": tid, "scale": scale,
                    "question": QUESTION_FAMILIES[q],
                    "confidence": round(mean_conf, 4),
                    "variance": round(v, 6),
                })

    # Correlation
    confs = [p["confidence"] for p in points]
    vars_ = [p["variance"] for p in points]
    if np.std(confs) > 0 and np.std(vars_) > 0:
        pearson_r, _ = sp_stats.pearsonr(confs, vars_)
        spearman_r, _ = sp_stats.spearmanr(confs, vars_)
    else:
        pearson_r, spearman_r = 0.0, 0.0

    return {
        "points": points,
        "pearsonR": round(float(pearson_r), 4),
        "spearmanR": round(float(spearman_r), 4),
    }


def compute_confidence_by_scale(logprobs, text_ids):
    """Mean confidence per scale."""
    rows = []
    for scale in SCALE_ORDER:
        probs = []
        for tid in text_ids:
            for sample_lps in logprobs[tid][scale]:
                for lp in sample_lps:
                    probs.append(lp.get("probability", 1.0))
        arr = np.array(probs)
        rows.append({
            "scale": scale,
            "meanConf": round(float(np.mean(arr)), 4),
            "medianConf": round(float(np.median(arr)), 4),
            "minConf": round(float(np.min(arr)), 4),
            "pctBelow90": round(float(np.mean(arr < 0.90)) * 100, 2),
            "pctBelow50": round(float(np.mean(arr < 0.50)) * 100, 2),
        })
    return rows


def compute_question_confidence_ranking(logprobs, text_ids):
    """Questions ranked by mean confidence on continuous scale."""
    rows = []
    for q_idx, family in enumerate(QUESTION_FAMILIES):
        probs = []
        for tid in text_ids:
            for sample_lps in logprobs[tid]["continuous"]:
                if q_idx < len(sample_lps):
                    probs.append(sample_lps[q_idx].get("probability", 1.0))
        arr = np.array(probs)
        rows.append({
            "question": QUESTION_LABELS[q_idx],
            "family": family,
            "meanConf": round(float(np.mean(arr)), 4),
            "pctBelow90": round(float(np.mean(arr < 0.90)) * 100, 1),
        })
    rows.sort(key=lambda x: x["meanConf"])
    return rows


def compute_confidence_distribution(logprobs, text_ids):
    """Histogram bins of logprob probabilities per scale."""
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.001]
    bin_labels = ["<50%", "50-70%", "70-80%", "80-90%", "90-95%", "95-99%", "99-100%"]
    result = {}
    for scale in SCALE_ORDER:
        probs = []
        for tid in text_ids:
            for sample_lps in logprobs[tid][scale]:
                for lp in sample_lps:
                    probs.append(lp.get("probability", 1.0))
        counts, _ = np.histogram(probs, bins=bins)
        total = len(probs)
        result[scale] = [
            {"bin": label, "count": int(c), "pct": round(float(c / total * 100), 1)}
            for label, c in zip(bin_labels, counts)
        ]
    return result


# ─── Section 4: Predictive Power ─────────────────────────────────────────────

def compute_predictive_power(s2, logprobs, text_ids):
    """
    Can 1 sample's logprobs predict multi-sample variance?
    Take first sample only. Flag logprob < threshold as 'unreliable'.
    Compare against ground-truth: variance > 0 from 20 samples.
    """
    # Collect data: one entry per (text, question, scale) = 800 points
    entries = []
    for tid in text_ids:
        for scale in SCALE_ORDER:
            for q in range(20):
                # Ground truth: is there any variance?
                v = float(np.var(s2[tid][scale][:, q], ddof=0))
                has_variance = v > 0

                # First sample's logprob confidence
                first_sample_lps = logprobs[tid][scale][0]
                if q < len(first_sample_lps):
                    conf = first_sample_lps[q].get("probability", 1.0)
                else:
                    conf = 1.0

                entries.append({
                    "text": tid, "scale": scale, "question": QUESTION_FAMILIES[q],
                    "confidence": conf, "variance": v, "hasVariance": has_variance,
                })

    # Sweep thresholds for ROC
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 1.0]
    roc_points = []
    best_f1 = 0
    best_threshold = 0.9
    best_metrics = {}

    for thresh in thresholds:
        tp, fp, tn, fn = 0, 0, 0, 0
        for e in entries:
            predicted_unstable = e["confidence"] < thresh
            actually_unstable = e["hasVariance"]
            if predicted_unstable and actually_unstable:
                tp += 1
            elif predicted_unstable and not actually_unstable:
                fp += 1
            elif not predicted_unstable and actually_unstable:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        accuracy = (tp + tn) / len(entries)

        point = {
            "threshold": thresh,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "tpr": round(tpr, 4),
            "fpr": round(fpr, 4),
        }
        roc_points.append(point)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
            best_metrics = point

    # AUC approximation (trapezoidal)
    roc_sorted = sorted(roc_points, key=lambda x: x["fpr"])
    auc = 0
    for i in range(1, len(roc_sorted)):
        dx = roc_sorted[i]["fpr"] - roc_sorted[i-1]["fpr"]
        dy = (roc_sorted[i]["tpr"] + roc_sorted[i-1]["tpr"]) / 2
        auc += dx * dy

    # Count actual positives/negatives
    n_unstable = sum(1 for e in entries if e["hasVariance"])
    n_stable = len(entries) - n_unstable

    return {
        "rocPoints": roc_points,
        "bestThreshold": best_threshold,
        "bestMetrics": best_metrics,
        "auc": round(auc, 4),
        "nTotal": len(entries),
        "nUnstable": n_unstable,
        "nStable": n_stable,
        "baseRate": round(n_unstable / len(entries) * 100, 1),
    }


# ─── Section 5: Score Agreement ──────────────────────────────────────────────

def compute_score_agreement(s1, s2, text_ids):
    """Compare modal scores between runs."""
    agree_count = 0
    total = 0
    per_scale = {}

    for scale in SCALE_ORDER:
        scale_agree = 0
        scale_total = 0
        for tid in text_ids:
            for q in range(20):
                col1 = s1[tid][scale][:, q]
                col2 = s2[tid][scale][:, q]
                # Modal score
                vals1, cnt1 = np.unique(col1, return_counts=True)
                vals2, cnt2 = np.unique(col2, return_counts=True)
                mode1 = vals1[np.argmax(cnt1)]
                mode2 = vals2[np.argmax(cnt2)]
                if abs(mode1 - mode2) < 1e-6:
                    agree_count += 1
                    scale_agree += 1
                total += 1
                scale_total += 1
        per_scale[scale] = {
            "agreePct": round(scale_agree / scale_total * 100, 1),
            "disagreeCount": scale_total - scale_agree,
            "total": scale_total,
        }

    return {
        "overallAgreePct": round(agree_count / total * 100, 1),
        "totalPairs": total,
        "perScale": per_scale,
    }


def compute_unique_vector_comparison(s1, s2, text_ids):
    """Unique vectors per text x scale, both runs."""
    rows = []
    for tid in text_ids:
        for scale in SCALE_ORDER:
            r1 = np.round(s1[tid][scale], 4)
            r2 = np.round(s2[tid][scale], 4)
            u1 = len(set(tuple(r) for r in r1))
            u2 = len(set(tuple(r) for r in r2))
            rows.append({
                "text": tid, "scale": scale,
                "run1Unique": u1, "run2Unique": u2,
                "delta": u2 - u1,
            })
    return rows


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading Run 1...")
    s1, text_ids = load_run1()
    print("Loading Run 2 (with logprobs)...")
    s2, logprobs, _ = load_run2()

    print("\n--- Section 1: Run-to-Run Reproducibility ---")
    scale_deltas = compute_scale_deltas(s1, s2, text_ids)
    for r in scale_deltas:
        print(f"  {r['scale']:12s}: Run1={r['run1MeanVar']:.6f} Run2={r['run2MeanVar']:.6f} delta={r['deltaPct']:+.1f}%")

    heatmap_diff = compute_heatmap_diff(s1, s2, text_ids)
    var_scatter = compute_variance_scatter(s1, s2, text_ids)
    print(f"  Variance scatter: Pearson r = {var_scatter['pearsonR']:.4f}")

    print("\n--- Section 3: Logprob Analysis ---")
    conf_heatmap = compute_confidence_heatmap(logprobs, text_ids)
    conf_vs_var = compute_confidence_vs_variance(s2, logprobs, text_ids)
    print(f"  Confidence vs Variance: Pearson r = {conf_vs_var['pearsonR']:.4f}, Spearman = {conf_vs_var['spearmanR']:.4f}")
    conf_by_scale = compute_confidence_by_scale(logprobs, text_ids)
    for r in conf_by_scale:
        print(f"  {r['scale']:12s}: mean={r['meanConf']:.4f} pct<90%={r['pctBelow90']}%")
    question_ranking = compute_question_confidence_ranking(logprobs, text_ids)
    conf_distribution = compute_confidence_distribution(logprobs, text_ids)

    print("\n--- Section 4: Predictive Power ---")
    predictive = compute_predictive_power(s2, logprobs, text_ids)
    print(f"  Base rate (unstable): {predictive['baseRate']}% ({predictive['nUnstable']}/{predictive['nTotal']})")
    bm = predictive["bestMetrics"]
    print(f"  Best threshold: {predictive['bestThreshold']}")
    print(f"  Precision={bm['precision']:.3f} Recall={bm['recall']:.3f} F1={bm['f1']:.3f} Accuracy={bm['accuracy']:.3f}")
    print(f"  AUC = {predictive['auc']:.4f}")

    print("\n--- Section 5: Score Agreement ---")
    agreement = compute_score_agreement(s1, s2, text_ids)
    print(f"  Overall modal agreement: {agreement['overallAgreePct']}%")
    for scale, data in agreement["perScale"].items():
        print(f"  {scale:12s}: {data['agreePct']}% agree ({data['disagreeCount']} disagree)")
    unique_vectors = compute_unique_vector_comparison(s1, s2, text_ids)

    # Save
    output = {
        "section1_reproducibility": {
            "scaleDeltas": scale_deltas,
            "heatmapDiff": heatmap_diff,
            "varianceScatter": var_scatter,
        },
        "section3_logprobs": {
            "confidenceHeatmap": conf_heatmap,
            "confidenceVsVariance": conf_vs_var,
            "confidenceByScale": conf_by_scale,
            "questionConfidenceRanking": question_ranking,
            "confidenceDistribution": conf_distribution,
        },
        "section4_predictive": predictive,
        "section5_agreement": {
            "modalAgreement": agreement,
            "uniqueVectors": unique_vectors,
        },
    }

    out_path = PROJECT_ROOT / "data" / "comparison_metrics.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
