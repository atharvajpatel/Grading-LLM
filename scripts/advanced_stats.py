"""
Advanced Statistical Analysis for Grading-LLM
==============================================
Computes all 3 tiers of reliability metrics from batch_results.json:

Tier 1 (Essential):
  - Cronbach's alpha (internal consistency per scale)
  - ICC (intraclass correlation coefficient)
  - Cohen's kappa (discrete scales) / Krippendorff's alpha (all scales)

Tier 2 (Informative):
  - Test-retest correlation (Pearson/Spearman between sample pairs)
  - Coefficient of Variation per question
  - Scale degradation regression (linear fit: does consistency degrade binary->continuous?)
  - Bland-Altman analysis (mean difference + limits of agreement between sample pairs)

Tier 3 (Nice to Have):
  - Bootstrap confidence intervals on variance/consistency
  - Friedman test (non-parametric: does scale granularity significantly affect consistency?)
  - Effect sizes (eta-squared: how much variance is explained by scale vs text vs question)

Reads: data/batch_results.json (10 texts x 20 questions x 4 scales x 20 samples)
Writes: data/advanced_stats_results.json
"""

import json
import sys
import os
from pathlib import Path
from itertools import combinations

import numpy as np
from scipy import stats as sp_stats
from sklearn.metrics import cohen_kappa_score

import krippendorff


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_data(path: str) -> dict:
    """Load batch_results.json and reshape into usable structures."""
    with open(path) as f:
        raw = json.load(f)

    SCALE_ORDER = ["binary", "ternary", "quaternary", "continuous"]
    QUESTION_FAMILIES = [
        "named_entities", "actions_events", "causality", "temporal", "spatial",
        "numeric", "negation", "uncertainty", "modality", "sentiment",
        "emotion", "social", "dialogue", "first_person", "imperative",
        "comparison", "normative", "intent", "concreteness", "identity"
    ]

    text_ids = [item["id"] for item in raw]

    # Build: scores[text_id][scale] = np.array of shape (20 samples, 20 questions)
    scores = {}
    for item in raw:
        tid = item["id"]
        scores[tid] = {}
        for scale in SCALE_ORDER:
            arr = np.array(item["raw_scores"][scale])  # (20, 20)
            scores[tid][scale] = arr

    return {
        "scores": scores,
        "text_ids": text_ids,
        "scale_order": SCALE_ORDER,
        "question_families": QUESTION_FAMILIES,
        "n_samples": 20,
        "n_questions": 20,
        "n_texts": len(text_ids),
    }


# ─── TIER 1: Essential Reliability Metrics ───────────────────────────────────

def cronbachs_alpha(data: np.ndarray) -> float:
    """
    Cronbach's alpha for internal consistency.
    data: (n_subjects, n_items) — rows are subjects, columns are items.
    For us: rows = samples (20), columns = questions (20).
    """
    n_items = data.shape[1]
    if n_items < 2:
        return float("nan")
    item_variances = np.var(data, axis=0, ddof=1)
    total_variance = np.var(np.sum(data, axis=1), ddof=1)
    if total_variance == 0:
        return 1.0  # perfect agreement
    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
    return float(alpha)


def icc_two_way_agreement(data: np.ndarray) -> float:
    """
    ICC(2,1) — two-way random, absolute agreement, single measures.
    data: (n_subjects, n_raters) — for us: (20 questions, 20 samples)
    Each question is a "subject", each sample is a "rater".
    """
    n, k = data.shape  # n=subjects, k=raters
    if n < 2 or k < 2:
        return float("nan")

    grand_mean = np.mean(data)
    row_means = np.mean(data, axis=1)
    col_means = np.mean(data, axis=0)

    # Sum of squares
    ss_total = np.sum((data - grand_mean) ** 2)
    ss_rows = k * np.sum((row_means - grand_mean) ** 2)  # between subjects
    ss_cols = n * np.sum((col_means - grand_mean) ** 2)  # between raters
    ss_error = ss_total - ss_rows - ss_cols

    # Mean squares
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0

    # ICC(2,1) formula
    denom = ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n
    if denom == 0:
        return 1.0
    icc = (ms_rows - ms_error) / denom
    return float(np.clip(icc, -1.0, 1.0))


def cohens_kappa_multi_rater(data: np.ndarray) -> float:
    """
    Average pairwise Cohen's kappa across all sample pairs.
    data: (n_samples, n_questions) with discrete values.
    """
    n_samples = data.shape[0]
    if n_samples < 2:
        return float("nan")

    kappas = []
    for i, j in combinations(range(n_samples), 2):
        # Skip if both columns are constant and identical
        if np.array_equal(data[i], data[j]):
            kappas.append(1.0)
            continue
        # Skip if either rater has zero variance (kappa undefined)
        if len(np.unique(data[i])) < 2 and len(np.unique(data[j])) < 2:
            kappas.append(1.0 if np.array_equal(data[i], data[j]) else 0.0)
            continue
        try:
            k = cohen_kappa_score(data[i], data[j])
            kappas.append(k)
        except Exception:
            kappas.append(float("nan"))

    valid = [k for k in kappas if not np.isnan(k)]
    return float(np.mean(valid)) if valid else float("nan")


def krippendorff_alpha(data: np.ndarray, level: str = "interval") -> float:
    """
    Krippendorff's alpha.
    data: (n_raters, n_units) — for us: (20 samples, 20 questions).
    level: 'nominal', 'ordinal', 'interval', 'ratio'
    """
    try:
        alpha = krippendorff.alpha(
            reliability_data=data,
            level_of_measurement=level
        )
        return float(alpha) if not np.isnan(alpha) else float("nan")
    except Exception:
        return float("nan")


def compute_tier1(info: dict) -> dict:
    """Compute all Tier 1 metrics."""
    scores = info["scores"]
    scale_order = info["scale_order"]
    text_ids = info["text_ids"]

    results = {
        "cronbachs_alpha": {},
        "icc": {},
        "cohens_kappa": {},
        "krippendorff_alpha": {},
    }

    for scale in scale_order:
        ca_values = []
        icc_values = []
        kappa_values = []
        kripp_values = []

        per_text = {}
        for tid in text_ids:
            data = scores[tid][scale]  # (20 samples, 20 questions)

            # Cronbach's alpha: treat samples as subjects, questions as items
            ca = cronbachs_alpha(data)
            ca_values.append(ca)

            # ICC(2,1): treat questions as subjects, samples as raters
            icc_val = icc_two_way_agreement(data.T)  # (20 questions, 20 samples)
            icc_values.append(icc_val)

            # Cohen's kappa (only for discrete scales)
            if scale != "continuous":
                kappa = cohens_kappa_multi_rater(data)
                kappa_values.append(kappa)
            else:
                kappa = None

            # Krippendorff's alpha
            level = "nominal" if scale == "binary" else (
                "ordinal" if scale in ["ternary", "quaternary"] else "interval"
            )
            kripp = krippendorff_alpha(data, level=level)
            kripp_values.append(kripp)

            per_text[tid] = {
                "cronbachs_alpha": round(ca, 6) if not np.isnan(ca) else None,
                "icc": round(icc_val, 6) if not np.isnan(icc_val) else None,
                "cohens_kappa": round(kappa, 6) if kappa is not None and not np.isnan(kappa) else None,
                "krippendorff_alpha": round(kripp, 6) if not np.isnan(kripp) else None,
            }

        # Aggregate
        valid_ca = [v for v in ca_values if not np.isnan(v)]
        valid_icc = [v for v in icc_values if not np.isnan(v)]
        valid_kappa = [v for v in kappa_values if v is not None and not np.isnan(v)]
        valid_kripp = [v for v in kripp_values if not np.isnan(v)]

        results["cronbachs_alpha"][scale] = {
            "mean": round(float(np.mean(valid_ca)), 6) if valid_ca else None,
            "min": round(float(np.min(valid_ca)), 6) if valid_ca else None,
            "max": round(float(np.max(valid_ca)), 6) if valid_ca else None,
            "per_text": {tid: per_text[tid]["cronbachs_alpha"] for tid in text_ids},
        }
        results["icc"][scale] = {
            "mean": round(float(np.mean(valid_icc)), 6) if valid_icc else None,
            "min": round(float(np.min(valid_icc)), 6) if valid_icc else None,
            "max": round(float(np.max(valid_icc)), 6) if valid_icc else None,
            "per_text": {tid: per_text[tid]["icc"] for tid in text_ids},
        }
        results["cohens_kappa"][scale] = {
            "mean": round(float(np.mean(valid_kappa)), 6) if valid_kappa else None,
            "min": round(float(np.min(valid_kappa)), 6) if valid_kappa else None,
            "max": round(float(np.max(valid_kappa)), 6) if valid_kappa else None,
            "per_text": {tid: per_text[tid]["cohens_kappa"] for tid in text_ids},
        } if scale != "continuous" else {
            "mean": None, "min": None, "max": None,
            "per_text": {tid: None for tid in text_ids},
            "note": "Cohen's kappa not applicable to continuous scale"
        }
        results["krippendorff_alpha"][scale] = {
            "mean": round(float(np.mean(valid_kripp)), 6) if valid_kripp else None,
            "min": round(float(np.min(valid_kripp)), 6) if valid_kripp else None,
            "max": round(float(np.max(valid_kripp)), 6) if valid_kripp else None,
            "per_text": {tid: per_text[tid]["krippendorff_alpha"] for tid in text_ids},
        }

    return results


# ─── TIER 2: Informative Metrics ─────────────────────────────────────────────

def test_retest_correlation(data: np.ndarray) -> dict:
    """
    Test-retest: split 20 samples into odd/even halves, correlate.
    data: (20 samples, 20 questions)
    Returns Pearson r, Spearman rho, and p-values.
    """
    odd = data[0::2]   # samples 0,2,4,...,18 → 10 samples
    even = data[1::2]  # samples 1,3,5,...,19 → 10 samples

    # Mean across the half for each question
    odd_means = np.mean(odd, axis=0)   # (20,)
    even_means = np.mean(even, axis=0)  # (20,)

    # If either half is constant, correlation is undefined
    if np.std(odd_means) == 0 or np.std(even_means) == 0:
        return {"pearson_r": 1.0, "pearson_p": 0.0, "spearman_rho": 1.0, "spearman_p": 0.0}

    pearson_r, pearson_p = sp_stats.pearsonr(odd_means, even_means)
    spearman_rho, spearman_p = sp_stats.spearmanr(odd_means, even_means)

    return {
        "pearson_r": round(float(pearson_r), 6),
        "pearson_p": round(float(pearson_p), 6),
        "spearman_rho": round(float(spearman_rho), 6),
        "spearman_p": round(float(spearman_p), 6),
    }


def coefficient_of_variation(data: np.ndarray) -> list:
    """
    CV = std/mean for each question across 20 samples.
    data: (20 samples, 20 questions)
    Returns list of 20 CV values.
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0, ddof=1)
    cvs = []
    for m, s in zip(means, stds):
        if m == 0:
            cvs.append(0.0 if s == 0 else float("inf"))
        else:
            cvs.append(float(s / abs(m)))
    return [round(v, 6) if v != float("inf") else "inf" for v in cvs]


def scale_degradation_regression(info: dict) -> dict:
    """
    Fit linear regression: x = scale index (0,1,2,3), y = mean variance.
    Tests if consistency linearly degrades from binary to continuous.
    """
    scores = info["scores"]
    scale_order = info["scale_order"]
    text_ids = info["text_ids"]

    # Compute mean variance per scale across all texts
    x = np.array([0, 1, 2, 3])  # binary=0, ternary=1, quaternary=2, continuous=3
    per_text_results = {}

    all_y = []
    for tid in text_ids:
        y = []
        for scale in scale_order:
            data = scores[tid][scale]  # (20, 20)
            variances = np.var(data, axis=0, ddof=0)  # variance per question
            y.append(float(np.mean(variances)))
        all_y.append(y)

        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, y)
        per_text_results[tid] = {
            "slope": round(float(slope), 8),
            "intercept": round(float(intercept), 8),
            "r_squared": round(float(r_value ** 2), 6),
            "p_value": round(float(p_value), 6),
            "mean_variances": [round(v, 6) for v in y],
        }

    # Overall
    overall_y = np.mean(all_y, axis=0)
    slope, intercept, r_value, p_value, std_err = sp_stats.linregress(x, overall_y.tolist())

    return {
        "overall": {
            "slope": round(float(slope), 8),
            "intercept": round(float(intercept), 8),
            "r_squared": round(float(r_value ** 2), 6),
            "p_value": round(float(p_value), 6),
            "mean_variances_by_scale": {
                scale: round(float(v), 6) for scale, v in zip(scale_order, overall_y)
            },
            "interpretation": (
                "positive slope = variance increases with scale granularity"
                if slope > 0 else
                "negative slope = variance decreases with scale granularity"
            ),
        },
        "per_text": per_text_results,
    }


def bland_altman(data: np.ndarray, n_pairs: int = 50) -> dict:
    """
    Bland-Altman analysis: compare random pairs of samples.
    data: (20 samples, 20 questions)
    For each pair of samples, compute mean and difference across all questions.
    """
    n_samples = data.shape[0]
    rng = np.random.RandomState(42)

    means_list = []
    diffs_list = []
    for _ in range(n_pairs):
        i, j = rng.choice(n_samples, 2, replace=False)
        pair_means = (data[i] + data[j]) / 2.0
        pair_diffs = data[i] - data[j]
        means_list.extend(pair_means.tolist())
        diffs_list.extend(pair_diffs.tolist())

    means_arr = np.array(means_list)
    diffs_arr = np.array(diffs_list)

    mean_diff = float(np.mean(diffs_arr))
    std_diff = float(np.std(diffs_arr, ddof=1))
    loa_upper = mean_diff + 1.96 * std_diff
    loa_lower = mean_diff - 1.96 * std_diff

    return {
        "mean_difference": round(mean_diff, 8),
        "std_difference": round(std_diff, 8),
        "limits_of_agreement": {
            "lower": round(loa_lower, 8),
            "upper": round(loa_upper, 8),
        },
        "n_comparisons": len(diffs_list),
        "pct_within_loa": round(
            float(np.mean((diffs_arr >= loa_lower) & (diffs_arr <= loa_upper))) * 100, 2
        ),
    }


def compute_tier2(info: dict) -> dict:
    """Compute all Tier 2 metrics."""
    scores = info["scores"]
    scale_order = info["scale_order"]
    text_ids = info["text_ids"]
    question_families = info["question_families"]

    results = {
        "test_retest_correlation": {},
        "coefficient_of_variation": {},
        "bland_altman": {},
    }

    for scale in scale_order:
        trt_values = {}
        cv_values = {}
        ba_values = {}

        for tid in text_ids:
            data = scores[tid][scale]
            trt_values[tid] = test_retest_correlation(data)
            cv_raw = coefficient_of_variation(data)
            cv_values[tid] = {
                "per_question": {
                    question_families[i]: cv_raw[i]
                    for i in range(len(question_families))
                },
                "mean_cv": round(
                    float(np.mean([v for v in cv_raw if v != "inf"])), 6
                ) if any(v != "inf" for v in cv_raw) else None,
            }
            ba_values[tid] = bland_altman(data)

        # Aggregate test-retest
        all_pearson = [v["pearson_r"] for v in trt_values.values()]
        all_spearman = [v["spearman_rho"] for v in trt_values.values()]
        results["test_retest_correlation"][scale] = {
            "mean_pearson_r": round(float(np.mean(all_pearson)), 6),
            "mean_spearman_rho": round(float(np.mean(all_spearman)), 6),
            "per_text": trt_values,
        }
        results["coefficient_of_variation"][scale] = {
            "per_text": cv_values,
        }

        # Aggregate Bland-Altman
        all_mean_diff = [v["mean_difference"] for v in ba_values.values()]
        all_std_diff = [v["std_difference"] for v in ba_values.values()]
        results["bland_altman"][scale] = {
            "mean_bias": round(float(np.mean(all_mean_diff)), 8),
            "mean_std_diff": round(float(np.mean(all_std_diff)), 8),
            "per_text": ba_values,
        }

    # Scale degradation regression
    results["scale_degradation_regression"] = scale_degradation_regression(info)

    return results


# ─── TIER 3: Advanced Metrics ────────────────────────────────────────────────

def bootstrap_ci(
    data: np.ndarray,
    stat_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> dict:
    """
    Bootstrap confidence interval for a statistic.
    data: 1D array
    stat_fn: function that computes the statistic on a 1D array
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        boot_stats.append(stat_fn(sample))

    boot_stats = np.array(boot_stats)
    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_stats, alpha * 100))
    upper = float(np.percentile(boot_stats, (1 - alpha) * 100))
    point = float(stat_fn(data))

    return {
        "point_estimate": round(point, 6),
        "ci_lower": round(lower, 6),
        "ci_upper": round(upper, 6),
        "ci_level": ci,
        "n_bootstrap": n_bootstrap,
    }


def friedman_test(info: dict) -> dict:
    """
    Friedman test: does scale granularity significantly affect mean variance?
    Treats each (text, question) pair as a subject,
    and the 4 scale variances as repeated measures.
    """
    scores = info["scores"]
    scale_order = info["scale_order"]
    text_ids = info["text_ids"]
    n_questions = info["n_questions"]

    # Build: rows = (text, question) pairs, columns = 4 scales
    rows = []
    for tid in text_ids:
        for q_idx in range(n_questions):
            row = []
            for scale in scale_order:
                data = scores[tid][scale][:, q_idx]  # 20 samples for this question
                row.append(float(np.var(data, ddof=0)))
            rows.append(row)

    matrix = np.array(rows)  # (200, 4)

    # Friedman test requires at least some variation
    try:
        stat, p_value = sp_stats.friedmanchisquare(
            matrix[:, 0], matrix[:, 1], matrix[:, 2], matrix[:, 3]
        )
    except Exception as e:
        return {"error": str(e)}

    return {
        "chi_squared": round(float(stat), 6),
        "p_value": float(p_value),
        "df": 3,
        "n_subjects": len(rows),
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
        "interpretation": (
            "Scale granularity significantly affects variance (p < 0.05)"
            if p_value < 0.05 else
            "No significant effect of scale granularity on variance"
        ),
    }


def eta_squared_analysis(info: dict) -> dict:
    """
    Eta-squared: how much variance in scores is explained by:
    - Scale type (binary/ternary/quaternary/continuous)
    - Text type (10 text categories)
    - Question type (20 question families)

    Uses one-way ANOVA for each factor on the per-cell variance values.
    """
    scores = info["scores"]
    scale_order = info["scale_order"]
    text_ids = info["text_ids"]
    n_questions = info["n_questions"]
    question_families = info["question_families"]

    # Collect all variance values with their factor labels
    variances = []
    scale_labels = []
    text_labels = []
    question_labels = []

    for tid in text_ids:
        for scale in scale_order:
            for q_idx in range(n_questions):
                data = scores[tid][scale][:, q_idx]
                v = float(np.var(data, ddof=0))
                variances.append(v)
                scale_labels.append(scale)
                text_labels.append(tid)
                question_labels.append(question_families[q_idx])

    variances = np.array(variances)

    def compute_eta_sq(labels, values):
        """One-way ANOVA eta-squared."""
        groups = {}
        for label, val in zip(labels, values):
            groups.setdefault(label, []).append(val)

        group_arrays = [np.array(g) for g in groups.values()]

        # Check we have multiple groups with variation
        if len(group_arrays) < 2:
            return {"eta_squared": 0.0, "f_statistic": 0.0, "p_value": 1.0}

        try:
            f_stat, p_value = sp_stats.f_oneway(*group_arrays)
        except Exception:
            return {"eta_squared": 0.0, "f_statistic": 0.0, "p_value": 1.0}

        # Eta-squared = SS_between / SS_total
        grand_mean = np.mean(values)
        ss_total = np.sum((values - grand_mean) ** 2)
        ss_between = sum(
            len(g) * (np.mean(g) - grand_mean) ** 2 for g in group_arrays
        )
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

        return {
            "eta_squared": round(float(eta_sq), 6),
            "f_statistic": round(float(f_stat), 4) if not np.isnan(f_stat) else None,
            "p_value": float(p_value) if not np.isnan(p_value) else None,
        }

    result = {
        "by_scale": compute_eta_sq(scale_labels, variances),
        "by_text": compute_eta_sq(text_labels, variances),
        "by_question": compute_eta_sq(question_labels, variances),
    }

    # Add interpretation
    factors = sorted(result.items(), key=lambda x: x[1]["eta_squared"], reverse=True)
    result["ranking"] = [
        {"factor": name, "eta_squared": vals["eta_squared"]}
        for name, vals in factors
        if name != "ranking"
    ]
    result["interpretation"] = (
        f"'{factors[0][0].replace('by_', '')}' explains the most variance "
        f"(eta^2 = {factors[0][1]['eta_squared']:.4f}), followed by "
        f"'{factors[1][0].replace('by_', '')}' ({factors[1][1]['eta_squared']:.4f}) "
        f"and '{factors[2][0].replace('by_', '')}' ({factors[2][1]['eta_squared']:.4f})"
    )

    return result


def compute_tier3(info: dict) -> dict:
    """Compute all Tier 3 metrics."""
    scores = info["scores"]
    scale_order = info["scale_order"]
    text_ids = info["text_ids"]
    n_questions = info["n_questions"]

    # Bootstrap CIs on mean variance and mean consistency per scale
    bootstrap_results = {}
    for scale in scale_order:
        # Collect all per-question variances across all texts
        all_variances = []
        all_consistencies = []
        for tid in text_ids:
            data = scores[tid][scale]  # (20, 20)
            for q_idx in range(n_questions):
                col = data[:, q_idx]
                all_variances.append(float(np.var(col, ddof=0)))
                # Mode consistency
                values, counts = np.unique(col, return_counts=True)
                all_consistencies.append(float(np.max(counts) / len(col)))

        var_arr = np.array(all_variances)
        con_arr = np.array(all_consistencies)

        bootstrap_results[scale] = {
            "variance": bootstrap_ci(var_arr, np.mean),
            "consistency": bootstrap_ci(con_arr, np.mean),
        }

    # Friedman test
    friedman = friedman_test(info)

    # Eta-squared
    eta_sq = eta_squared_analysis(info)

    return {
        "bootstrap_confidence_intervals": bootstrap_results,
        "friedman_test": friedman,
        "effect_sizes_eta_squared": eta_sq,
    }


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "data" / "batch_results.json"
    output_path = project_root / "data" / "advanced_stats_results.json"

    if not data_path.exists():
        print(f"ERROR: {data_path} not found")
        sys.exit(1)

    print(f"Loading data from {data_path}...")
    info = load_data(str(data_path))
    print(f"  {info['n_texts']} texts x {info['n_questions']} questions x "
          f"{len(info['scale_order'])} scales x {info['n_samples']} samples "
          f"= {info['n_texts'] * info['n_questions'] * len(info['scale_order']) * info['n_samples']:,} evaluations")

    print("\n--- TIER 1: Essential Reliability Metrics ---")
    tier1 = compute_tier1(info)
    for metric in ["cronbachs_alpha", "icc", "cohens_kappa", "krippendorff_alpha"]:
        print(f"  {metric}:")
        for scale in info["scale_order"]:
            val = tier1[metric][scale]["mean"]
            if val is not None:
                print(f"    {scale:12s}: {val:.4f}")
            else:
                print(f"    {scale:12s}: N/A")

    print("\n--- TIER 2: Informative Metrics ---")
    tier2 = compute_tier2(info)
    print("  Test-Retest Correlation:")
    for scale in info["scale_order"]:
        r = tier2["test_retest_correlation"][scale]["mean_pearson_r"]
        rho = tier2["test_retest_correlation"][scale]["mean_spearman_rho"]
        print(f"    {scale:12s}: Pearson r={r:.4f}, Spearman rho={rho:.4f}")

    print("  Scale Degradation Regression (overall):")
    reg = tier2["scale_degradation_regression"]["overall"]
    print(f"    slope={reg['slope']:.6f}, R²={reg['r_squared']:.4f}, p={reg['p_value']:.4f}")
    print(f"    {reg['interpretation']}")

    print("  Bland-Altman:")
    for scale in info["scale_order"]:
        ba = tier2["bland_altman"][scale]
        print(f"    {scale:12s}: bias={ba['mean_bias']:.6f}, std={ba['mean_std_diff']:.6f}")

    print("\n--- TIER 3: Advanced Metrics ---")
    tier3 = compute_tier3(info)
    print("  Bootstrap 95% CIs (mean variance):")
    for scale in info["scale_order"]:
        bv = tier3["bootstrap_confidence_intervals"][scale]["variance"]
        print(f"    {scale:12s}: {bv['point_estimate']:.6f} [{bv['ci_lower']:.6f}, {bv['ci_upper']:.6f}]")

    fr = tier3["friedman_test"]
    print(f"  Friedman Test: chi²={fr['chi_squared']:.2f}, p={fr['p_value']:.2e}, "
          f"significant={fr['significant_at_005']}")

    eta = tier3["effect_sizes_eta_squared"]
    print("  Eta-squared (% variance explained):")
    for item in eta["ranking"]:
        print(f"    {item['factor']:12s}: eta²={item['eta_squared']:.4f}")

    # Assemble final output
    output = {
        "metadata": {
            "source": "data/batch_results.json",
            "n_texts": info["n_texts"],
            "n_questions": info["n_questions"],
            "n_scales": len(info["scale_order"]),
            "n_samples": info["n_samples"],
            "total_evaluations": (
                info["n_texts"] * info["n_questions"]
                * len(info["scale_order"]) * info["n_samples"]
            ),
            "scales": info["scale_order"],
            "text_ids": info["text_ids"],
            "question_families": info["question_families"],
        },
        "tier1_essential": tier1,
        "tier2_informative": tier2,
        "tier3_advanced": tier3,
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults written to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
