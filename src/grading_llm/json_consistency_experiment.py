"""
JSON Mode Consistency Experiment

For each of 10 texts x 4 scales x 5 repeats:
1. Send all 20 questions in a single prompt with json_mode enabled
2. Request top-5 logprobs to measure confidence
3. Compare raw JSON outputs byte-for-byte
4. Compare parsed score arrays
5. Extract token-level confidence from logprobs
"""

import asyncio
import json
import time
import math
from pathlib import Path
from openai import AsyncOpenAI
from .config import get_api_key_for_local_dev
from .scales import format_scale_instruction, SCALE_ORDER

N_REPEATS = 5
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
DATA_DIR = Path(__file__).parent.parent.parent / "data"


def build_prompt(text: str, questions: list, scale_name: str) -> str:
    scale_instruction = format_scale_instruction(scale_name)
    q_lines = "\n".join(f'{i+1}. {q["question"]}' for i, q in enumerate(questions))
    n = len(questions)
    return f"""You are analyzing a statement for specific linguistic features. For each question, score how DEEPLY and RELEVANTLY the feature appears in the statement.

{scale_instruction}

STATEMENT TO ANALYZE:
"{text}"

For each question, consider:
- Is the feature explicitly stated or merely implied?
- Is it central to the statement's meaning or just peripheral/tangential?
- How strongly is it expressed?

Questions:
{q_lines}

Respond with ONLY a JSON object containing an array of scores:
{{"scores": [<score_1>, <score_2>, ..., <score_{n}>]}}"""


def extract_logprob_confidence(logprobs_content):
    """Extract confidence metrics from logprobs.

    Returns per-token data for score tokens and aggregate stats.
    """
    if not logprobs_content:
        return None

    tokens = logprobs_content
    score_tokens = []

    for tok in tokens:
        # Score tokens are the numeric values inside the scores array
        # Skip structural tokens like {, }, "scores", :, [, ], ,
        text = tok.token.strip()
        if not text:
            continue

        # Check if this is a numeric token (a score value)
        try:
            float(text)
        except ValueError:
            continue

        top_probs = {}
        if tok.top_logprobs:
            for alt in tok.top_logprobs:
                top_probs[alt.token] = math.exp(alt.logprob)

        chosen_prob = math.exp(tok.logprob)
        # Entropy over top-k tokens
        probs = [math.exp(alt.logprob) for alt in tok.top_logprobs] if tok.top_logprobs else [chosen_prob]
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)

        # Confidence gap: chosen prob minus second-best
        sorted_probs = sorted(probs, reverse=True)
        gap = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 1.0

        score_tokens.append({
            "token": text,
            "chosen_prob": round(chosen_prob, 6),
            "entropy": round(entropy, 6),
            "confidence_gap": round(gap, 6),
            "top_alternatives": {k: round(v, 6) for k, v in sorted(top_probs.items(), key=lambda x: -x[1])[:5]},
        })

    if not score_tokens:
        return None

    return {
        "n_score_tokens": len(score_tokens),
        "mean_chosen_prob": round(sum(t["chosen_prob"] for t in score_tokens) / len(score_tokens), 6),
        "mean_entropy": round(sum(t["entropy"] for t in score_tokens) / len(score_tokens), 6),
        "mean_confidence_gap": round(sum(t["confidence_gap"] for t in score_tokens) / len(score_tokens), 6),
        "min_chosen_prob": round(min(t["chosen_prob"] for t in score_tokens), 6),
        "per_token": score_tokens,
    }


async def run_experiment():
    client = AsyncOpenAI(api_key=get_api_key_for_local_dev())

    # Load data
    with open(DATA_DIR / "questions_mech.json") as f:
        questions = json.load(f)["questions"]
    with open(DATA_DIR / "batch_input_10.jsonl") as f:
        texts = [json.loads(line) for line in f]

    print(f"Texts: {len(texts)}, Questions: {len(questions)}, Scales: {len(SCALE_ORDER)}, Repeats: {N_REPEATS}")
    total_calls = len(texts) * len(SCALE_ORDER) * N_REPEATS
    print(f"Total API calls: {total_calls}")
    print()

    results = {}
    call_count = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    for text_obj in texts:
        text_id = text_obj["id"]
        text = text_obj["text"]
        results[text_id] = {}

        for scale in SCALE_ORDER:
            prompt = build_prompt(text, questions, scale)
            raw_jsons = []
            parsed_scores = []
            confidence_data = []

            for rep in range(N_REPEATS):
                call_count += 1
                print(f"  [{call_count}/{total_calls}] {text_id} / {scale} / rep {rep+1}", end="")

                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=TEMPERATURE,
                    max_tokens=500,
                    response_format={"type": "json_object"},
                    logprobs=True,
                    top_logprobs=5,
                )

                content = response.choices[0].message.content.strip()
                raw_jsons.append(content)

                # Track tokens
                if response.usage:
                    total_input_tokens += response.usage.prompt_tokens
                    total_output_tokens += response.usage.completion_tokens

                # Parse scores
                try:
                    data = json.loads(content)
                    scores = data.get("scores", [])
                    parsed_scores.append(scores)
                except json.JSONDecodeError:
                    parsed_scores.append(None)

                # Extract confidence from logprobs
                logprobs_content = None
                if response.choices[0].logprobs and response.choices[0].logprobs.content:
                    logprobs_content = response.choices[0].logprobs.content
                conf = extract_logprob_confidence(logprobs_content)
                confidence_data.append(conf)

                prob_str = f" (mean_prob={conf['mean_chosen_prob']:.3f}, min={conf['min_chosen_prob']:.3f})" if conf else ""
                print(prob_str)

            # Analysis: byte-for-byte comparison
            unique_jsons = list(set(raw_jsons))
            all_identical = len(unique_jsons) == 1

            # Analysis: score-level comparison
            valid_scores = [s for s in parsed_scores if s is not None]
            scores_identical = len(set(json.dumps(s) for s in valid_scores)) == 1 if valid_scores else False

            # Per-question variance across repeats
            per_question_variance = []
            if len(valid_scores) >= 2:
                for q_idx in range(len(questions)):
                    vals = [s[q_idx] for s in valid_scores if q_idx < len(s)]
                    if vals:
                        mean_val = sum(vals) / len(vals)
                        var_val = sum((v - mean_val) ** 2 for v in vals) / len(vals)
                        per_question_variance.append(round(var_val, 8))
                    else:
                        per_question_variance.append(None)

            # Aggregate confidence across repeats
            valid_conf = [c for c in confidence_data if c is not None]
            agg_confidence = None
            if valid_conf:
                agg_confidence = {
                    "mean_chosen_prob": round(sum(c["mean_chosen_prob"] for c in valid_conf) / len(valid_conf), 6),
                    "mean_entropy": round(sum(c["mean_entropy"] for c in valid_conf) / len(valid_conf), 6),
                    "mean_confidence_gap": round(sum(c["mean_confidence_gap"] for c in valid_conf) / len(valid_conf), 6),
                    "min_chosen_prob_across_reps": round(min(c["min_chosen_prob"] for c in valid_conf), 6),
                }

            results[text_id][scale] = {
                "raw_jsons_identical": all_identical,
                "n_unique_jsons": len(unique_jsons),
                "scores_identical": scores_identical,
                "per_question_variance": per_question_variance,
                "mean_variance": round(sum(v for v in per_question_variance if v is not None) / max(len(per_question_variance), 1), 8) if per_question_variance else 0,
                "confidence": agg_confidence,
                "raw_jsons": raw_jsons,
                "parsed_scores": valid_scores,
                "per_rep_confidence": [c for c in confidence_data if c is not None],
            }

    elapsed = time.time() - start_time

    # Build summary
    summary = {
        "metadata": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "n_texts": len(texts),
            "n_questions": len(questions),
            "n_scales": len(SCALE_ORDER),
            "n_repeats": N_REPEATS,
            "total_api_calls": total_calls,
            "json_mode": True,
            "logprobs": True,
            "top_logprobs": 5,
            "duration_seconds": round(elapsed, 1),
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
        },
        "summary": build_summary_table(results, texts, questions),
        "per_text": results,
    }

    out_path = DATA_DIR / "json_consistency_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s. Saved to {out_path}")
    print(f"Tokens: {total_input_tokens + total_output_tokens:,} (${(total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1e6:.4f})")
    print()
    print_summary(summary)


def build_summary_table(results, texts, questions):
    """Build a compact summary table."""
    table = []
    for scale in SCALE_ORDER:
        n_identical = 0
        n_total = 0
        all_variances = []
        all_probs = []
        all_entropies = []
        all_min_probs = []

        for text_obj in texts:
            tid = text_obj["id"]
            r = results[tid][scale]
            n_total += 1
            if r["raw_jsons_identical"]:
                n_identical += 1
            all_variances.append(r["mean_variance"])
            if r["confidence"]:
                all_probs.append(r["confidence"]["mean_chosen_prob"])
                all_entropies.append(r["confidence"]["mean_entropy"])
                all_min_probs.append(r["confidence"]["min_chosen_prob_across_reps"])

        table.append({
            "scale": scale,
            "pct_identical_jsons": round(n_identical / n_total * 100, 1),
            "n_identical": n_identical,
            "n_total": n_total,
            "mean_variance": round(sum(all_variances) / len(all_variances), 8) if all_variances else 0,
            "mean_token_prob": round(sum(all_probs) / len(all_probs), 4) if all_probs else None,
            "mean_token_entropy": round(sum(all_entropies) / len(all_entropies), 4) if all_entropies else None,
            "min_token_prob": round(min(all_min_probs), 4) if all_min_probs else None,
        })
    return table


def print_summary(summary):
    print("=" * 90)
    print("JSON MODE CONSISTENCY RESULTS")
    print("=" * 90)
    print(f"{'Scale':<14} {'Identical JSONs':>16} {'Mean Var':>12} {'Mean Prob':>12} {'Mean Entropy':>14} {'Min Prob':>10}")
    print("-" * 90)
    for row in summary["summary"]:
        ident = f"{row['n_identical']}/{row['n_total']} ({row['pct_identical_jsons']}%)"
        prob = f"{row['mean_token_prob']:.4f}" if row['mean_token_prob'] else "N/A"
        ent = f"{row['mean_token_entropy']:.4f}" if row['mean_token_entropy'] else "N/A"
        minp = f"{row['min_token_prob']:.4f}" if row['min_token_prob'] else "N/A"
        print(f"{row['scale']:<14} {ident:>16} {row['mean_variance']:>12.8f} {prob:>12} {ent:>14} {minp:>10}")
    print("=" * 90)

    # Per-text breakdown for non-identical cases
    print("\nNon-identical cases:")
    found_any = False
    for tid, scales in summary["per_text"].items():
        for scale, r in scales.items():
            if not r["raw_jsons_identical"]:
                found_any = True
                conf_str = ""
                if r["confidence"]:
                    conf_str = f" | min_prob={r['confidence']['min_chosen_prob_across_reps']:.4f}"
                print(f"  {tid} / {scale}: {r['n_unique_jsons']} unique JSONs, mean_var={r['mean_variance']:.6f}{conf_str}")
                # Show which questions differ
                if r["per_question_variance"]:
                    varying = [(i, v) for i, v in enumerate(r["per_question_variance"]) if v and v > 0]
                    if varying:
                        for qi, qv in varying:
                            print(f"    Q{qi+1} ({summary['per_text'][tid][scale].get('question_families', [''])[qi] if False else f'idx {qi}'}): var={qv:.6f}")
    if not found_any:
        print("  None! All JSONs were byte-for-byte identical across all 5 repeats.")


if __name__ == "__main__":
    asyncio.run(run_experiment())
