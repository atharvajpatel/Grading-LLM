"""
Batch Run with Logprobs
========================
Reruns the full 16,000 evaluation experiment (10 texts x 20 questions x 4 scales x 20 samples)
with OpenAI logprobs enabled on every API call.

For each score token the model outputs, we capture:
  - The logprob (log-probability) of the chosen token
  - The probability (exp(logprob)) as a 0-1 confidence
  - The top-5 alternative tokens and their logprobs

Output: data/batch_results_with_logprobs.json

Usage:
    python scripts/run_batch_with_logprobs.py
    python scripts/run_batch_with_logprobs.py --samples 5   # quick test with 5 samples
"""

import asyncio
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI, RateLimitError, APIError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

# ─── Config ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
SCALE_ORDER = ["binary", "ternary", "quaternary", "continuous"]
MODEL = "gpt-4o-mini"
TEMPERATURE = 0.0
MAX_CONCURRENT = 10
BATCH_SIZE = 20  # all 20 questions in one call
TOP_LOGPROBS = 5

# Pricing (per token)
PRICING = {
    "input": 0.150 / 1_000_000,
    "output": 0.600 / 1_000_000,
}

# Scale instructions (duplicated here to keep script self-contained)
SCALE_INSTRUCTIONS = {
    "binary": (
        "SCORING SCALE - Score based on how DEEPLY and RELEVANTLY the feature appears:\n"
        "  - 0 = Absent - not present in the statement\n"
        "  - 1 = Present - appears in the statement"
    ),
    "ternary": (
        "SCORING SCALE - Score based on how DEEPLY and RELEVANTLY the feature appears:\n"
        "  - 0 = Absent - not present at all\n"
        "  - 0.5 = Peripheral - present but minor, tangential, or implicit\n"
        "  - 1 = Central - clearly and explicitly present\n\n"
        "Use intermediate values (0.5) when the feature is:\n"
        "  - Present but peripheral/tangential to the main point\n"
        "  - Implicit or suggested rather than explicit\n"
        "  - Weakly expressed or only partially applicable"
    ),
    "quaternary": (
        "SCORING SCALE - Score based on how DEEPLY and RELEVANTLY the feature appears:\n"
        "  - 0 = Absent - not present\n"
        "  - 0.33 = Weak - barely present, very minor or implicit\n"
        "  - 0.66 = Moderate - present but not emphasized\n"
        "  - 1 = Strong - clearly present and significant\n\n"
        "Use intermediate values (0.33, 0.66) when the feature is:\n"
        "  - Present but peripheral/tangential to the main point\n"
        "  - Implicit or suggested rather than explicit\n"
        "  - Weakly expressed or only partially applicable"
    ),
    "continuous": (
        "SCORING SCALE - Rate how STRONGLY this feature appears on a continuous 0-1 scale:\n"
        "  - 0.0 = completely absent, no trace of this feature\n"
        "  - 0.1-0.3 = barely present, very minor or tangential\n"
        "  - 0.4-0.6 = moderately present, somewhat relevant but not central\n"
        "  - 0.7-0.9 = strongly present, clearly relevant and significant\n"
        "  - 1.0 = deeply and centrally present, core to the statement\n\n"
        "You may use ANY decimal value between 0.0 and 1.0 (e.g., 0.15, 0.42, 0.87).\n"
        "This is a CONTINUOUS scale - use the full range, not just 0 or 1."
    ),
}

VALID_SCALE_VALUES = {
    "binary": [0, 1],
    "ternary": [0, 0.5, 1],
    "quaternary": [0, 0.33, 0.66, 1],
    "continuous": None,  # any value 0-1
}


# ─── Usage Tracking ──────────────────────────────────────────────────────────

@dataclass
class UsageStats:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0

    def add(self, inp: int, out: int):
        self.input_tokens += inp
        self.output_tokens += out
        self.total_tokens += inp + out
        self.api_calls += 1
        self.cost_usd += inp * PRICING["input"] + out * PRICING["output"]


# ─── Prompt Construction ─────────────────────────────────────────────────────

def build_prompt(statement: str, questions: List[Dict], scale_name: str) -> str:
    """Build the exact same prompt format as the original run."""
    scale_instruction = SCALE_INSTRUCTIONS[scale_name]
    question_lines = []
    for i, q in enumerate(questions, 1):
        question_lines.append(f"{i}. {q['question']}")
    questions_block = "\n".join(question_lines)
    n = len(question_lines)

    return (
        f'You are analyzing a statement for specific linguistic features. '
        f'For each question, score how DEEPLY and RELEVANTLY the feature appears in the statement.\n\n'
        f'{scale_instruction}\n\n'
        f'STATEMENT TO ANALYZE:\n'
        f'"{statement}"\n\n'
        f'For each question, consider:\n'
        f'- Is the feature explicitly stated or merely implied?\n'
        f'- Is it central to the statement\'s meaning or just peripheral/tangential?\n'
        f'- How strongly is it expressed?\n\n'
        f'Questions:\n'
        f'{questions_block}\n\n'
        f'Respond with ONLY a JSON object containing an array of scores:\n'
        f'{{"scores": [<score_1>, <score_2>, ..., <score_{n}>]}}'
    )


# ─── Score Parsing ────────────────────────────────────────────────────────────

def validate_score(value: float, scale_name: str) -> float:
    """Snap/clamp a score to the valid range for the scale."""
    valid = VALID_SCALE_VALUES[scale_name]
    if valid is None:
        return max(0.0, min(1.0, value))
    return min(valid, key=lambda x: abs(x - value))


def parse_scores(content: str, n_questions: int, scale_name: str) -> List[float]:
    """Parse scores from model response text."""
    scores = []

    # Try JSON
    try:
        data = json.loads(content)
        if "scores" in data and isinstance(data["scores"], list):
            for s in data["scores"]:
                scores.append(validate_score(float(s), scale_name))
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback: regex for array
    if not scores:
        match = re.search(r'"scores"\s*:\s*\[([\d.,\s]+)\]', content)
        if match:
            nums = re.findall(r'[\d.]+', match.group(1))
            for n in nums:
                scores.append(validate_score(float(n), scale_name))

    # Fallback: find all numbers
    if not scores:
        nums = re.findall(r'(?<!["\w])(\d+\.?\d*)(?!["\w])', content)
        for n in nums[:n_questions]:
            scores.append(validate_score(float(n), scale_name))

    # Pad with zeros if needed
    while len(scores) < n_questions:
        scores.append(0.0)

    return scores[:n_questions]


# ─── Logprob Extraction ──────────────────────────────────────────────────────

def extract_score_logprobs(
    logprobs_content: List[Any],
    n_questions: int,
) -> List[Dict]:
    """
    Extract logprob data for each score token in the response.

    The model outputs something like: {"scores": [0, 1, 0, 0.5, ...]}
    We need to find the numeric tokens that correspond to actual score values
    and extract their logprobs + top alternatives.

    Returns a list of n_questions dicts, each with:
      - token: the actual token text
      - logprob: log-probability of chosen token
      - probability: exp(logprob) as 0-1 confidence
      - top_alternatives: list of {token, logprob, probability} for top-K alternatives
    """
    if not logprobs_content:
        return [_empty_logprob_entry() for _ in range(n_questions)]

    # Walk through tokens and find numeric score tokens
    # Strategy: find tokens after we've seen "scores" and "[", then collect numbers
    score_entries = []
    in_array = False
    bracket_depth = 0

    for tok_obj in logprobs_content:
        token_text = tok_obj.token.strip()

        # Track when we enter the scores array
        if "[" in tok_obj.token:
            in_array = True
            bracket_depth += 1
            continue
        if "]" in tok_obj.token:
            bracket_depth -= 1
            if bracket_depth <= 0:
                in_array = False
            continue

        if not in_array:
            continue

        # Skip commas, spaces, structural tokens
        if token_text in [",", "", " ", "{"  , "}", ":", '"', "'", "scores"]:
            continue

        # Check if this token looks numeric (could be part of a number like "0", ".5", "0.33")
        # We need to handle multi-token numbers like "0" + "." + "33"
        # OpenAI tokenizer usually keeps small numbers as single tokens though
        try:
            float(token_text)
            is_numeric = True
        except ValueError:
            # Could be a partial like "." -- skip
            is_numeric = False

        if is_numeric:
            # Extract top alternatives
            alternatives = []
            if tok_obj.top_logprobs:
                for alt in tok_obj.top_logprobs:
                    alternatives.append({
                        "token": alt.token.strip(),
                        "logprob": round(float(alt.logprob), 6),
                        "probability": round(math.exp(float(alt.logprob)), 6),
                    })

            score_entries.append({
                "token": token_text,
                "logprob": round(float(tok_obj.logprob), 6),
                "probability": round(math.exp(float(tok_obj.logprob)), 6),
                "top_alternatives": alternatives,
            })

    # If we got fewer entries than questions, pad with empties
    while len(score_entries) < n_questions:
        score_entries.append(_empty_logprob_entry())

    # If we got more (e.g., multi-token decimals), we need to merge them
    # Handle case where "0" and ".33" are separate tokens -> merge into one entry
    merged = _merge_multitoken_scores(score_entries, n_questions)

    return merged[:n_questions]


def _merge_multitoken_scores(entries: List[Dict], n_expected: int) -> List[Dict]:
    """
    Merge multi-token numbers. E.g., tokens "0", ".", "33" should become one entry for "0.33".
    When merging, use the logprob of the FIRST significant token (the integer part)
    since that's where the model makes the primary decision.
    """
    if len(entries) <= n_expected:
        return entries

    merged = []
    i = 0
    while i < len(entries):
        entry = entries[i]
        token = entry["token"]

        # Check if next tokens form a decimal: current="0", next=".", next_next="33"
        # or current="0", next=".33" or current="0.33" (already complete)
        if i + 1 < len(entries) and entries[i + 1]["token"].startswith("."):
            # Merge: take logprob from the integer part (primary decision)
            decimal_part = entries[i + 1]["token"]
            merged_token = token + decimal_part
            merged.append({
                "token": merged_token,
                "logprob": entry["logprob"],  # primary decision
                "probability": entry["probability"],
                "top_alternatives": entry["top_alternatives"],
            })
            i += 2

            # Check if there's more (e.g., "." then "33" as separate tokens)
            # Actually ".33" is usually one token, but let's be safe
            continue
        elif i + 2 < len(entries) and entries[i + 1]["token"] == ".":
            decimal_part = "." + entries[i + 2]["token"]
            merged_token = token + decimal_part
            merged.append({
                "token": merged_token,
                "logprob": entry["logprob"],
                "probability": entry["probability"],
                "top_alternatives": entry["top_alternatives"],
            })
            i += 3
            continue

        merged.append(entry)
        i += 1

    return merged


def _empty_logprob_entry() -> Dict:
    return {
        "token": "",
        "logprob": 0.0,
        "probability": 1.0,
        "top_alternatives": [],
    }


# ─── API Client ──────────────────────────────────────────────────────────────

class LogprobClient:
    def __init__(self, api_key: str):
        self.client = AsyncOpenAI(api_key=api_key)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self.usage = UsageStats()

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        retry=retry_if_exception_type((RateLimitError, APIError, ConnectionError)),
    )
    async def call(self, prompt: str) -> Tuple[str, List[Any]]:
        """Make API call with logprobs enabled. Returns (content, logprobs_content)."""
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=1000,
                logprobs=True,
                top_logprobs=TOP_LOGPROBS,
            )

            if response.usage:
                self.usage.add(response.usage.prompt_tokens, response.usage.completion_tokens)

            content = response.choices[0].message.content.strip()
            lp_content = []
            if response.choices[0].logprobs and response.choices[0].logprobs.content:
                lp_content = response.choices[0].logprobs.content

            return content, lp_content


# ─── Main Runner ─────────────────────────────────────────────────────────────

async def run_experiment(n_samples: int = 20):
    """Run the full experiment with logprobs."""
    # Load API key
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: No OPENAI_API_KEY found in .env or environment")
        sys.exit(1)

    # Load inputs
    texts_path = PROJECT_ROOT / "data" / "batch_input_10.jsonl"
    questions_path = PROJECT_ROOT / "data" / "questions_mech.json"

    texts = []
    with open(texts_path) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line))

    with open(questions_path) as f:
        questions_data = json.load(f)
    questions = questions_data["questions"]

    n_texts = len(texts)
    n_questions = len(questions)
    n_scales = len(SCALE_ORDER)
    total_calls = n_texts * n_scales * n_samples
    total_evals = n_texts * n_questions * n_scales * n_samples

    print(f"Experiment Configuration:")
    print(f"  Texts:     {n_texts}")
    print(f"  Questions: {n_questions}")
    print(f"  Scales:    {n_scales} ({', '.join(SCALE_ORDER)})")
    print(f"  Samples:   {n_samples}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Total evaluations: {total_evals:,}")
    print(f"  Model: {MODEL}, Temperature: {TEMPERATURE}")
    print(f"  Logprobs: top-{TOP_LOGPROBS}")
    print()

    client = LogprobClient(api_key)
    results = []
    completed_calls = 0
    start_time = time.time()

    for text_idx, text_item in enumerate(texts):
        tid = text_item["id"]
        statement = text_item["text"]

        text_result = {
            "id": tid,
            "text": statement,
            "raw_scores": {},
            "logprobs": {},
        }

        for scale in SCALE_ORDER:
            scale_scores = []       # List of 20 lists (each 20 scores)
            scale_logprobs = []     # List of 20 lists (each 20 logprob entries)

            for sample_idx in range(n_samples):
                prompt = build_prompt(statement, questions, scale)
                content, lp_content = await client.call(prompt)

                # Parse scores
                scores = parse_scores(content, n_questions, scale)

                # Extract logprobs for score tokens
                score_lps = extract_score_logprobs(lp_content, n_questions)

                scale_scores.append(scores)
                scale_logprobs.append(score_lps)

                completed_calls += 1

                # Progress
                elapsed = time.time() - start_time
                rate = completed_calls / elapsed if elapsed > 0 else 0
                eta = (total_calls - completed_calls) / rate if rate > 0 else 0
                pct = completed_calls / total_calls * 100

                sys.stdout.write(
                    f"\r  [{pct:5.1f}%] {tid} / {scale} / sample {sample_idx+1:2d}/{n_samples}"
                    f"  |  {completed_calls}/{total_calls} calls"
                    f"  |  {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining"
                    f"  |  ${client.usage.cost_usd:.4f}"
                )
                sys.stdout.flush()

            text_result["raw_scores"][scale] = scale_scores
            text_result["logprobs"][scale] = scale_logprobs

        print()  # newline after text completes
        results.append(text_result)

    # Final stats
    elapsed = time.time() - start_time
    print(f"\nExperiment complete!")
    print(f"  Duration:    {elapsed:.1f}s ({elapsed/60:.1f}m)")
    print(f"  API calls:   {client.usage.api_calls}")
    print(f"  Tokens:      {client.usage.total_tokens:,} "
          f"(in: {client.usage.input_tokens:,}, out: {client.usage.output_tokens:,})")
    print(f"  Cost:        ${client.usage.cost_usd:.4f}")

    # Compute summary statistics on logprob confidences
    summary = compute_logprob_summary(results, questions)

    # Build output
    output = {
        "metadata": {
            "model": MODEL,
            "temperature": TEMPERATURE,
            "n_texts": n_texts,
            "n_questions": n_questions,
            "n_scales": n_scales,
            "n_samples": n_samples,
            "total_evaluations": total_evals,
            "total_api_calls": client.usage.api_calls,
            "top_logprobs": TOP_LOGPROBS,
            "duration_seconds": round(elapsed, 1),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "usage": {
            "input_tokens": client.usage.input_tokens,
            "output_tokens": client.usage.output_tokens,
            "total_tokens": client.usage.total_tokens,
            "cost_usd": round(client.usage.cost_usd, 6),
        },
        "texts": results,
        "logprob_summary": summary,
    }

    # Save
    output_path = PROJECT_ROOT / "data" / "batch_results_with_logprobs.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


def compute_logprob_summary(results: List[Dict], questions: List[Dict]) -> Dict:
    """
    Compute aggregate statistics on logprob confidences across the experiment.
    """
    question_families = [q["family"] for q in questions]
    summary = {}

    for scale in SCALE_ORDER:
        all_probs = []           # flat list of all probabilities
        per_question_probs = [[] for _ in range(len(questions))]
        per_text_probs = {}

        for text_result in results:
            tid = text_result["id"]
            text_probs = []
            scale_logprobs = text_result["logprobs"][scale]

            for sample_lps in scale_logprobs:
                for q_idx, lp_entry in enumerate(sample_lps):
                    prob = lp_entry.get("probability", 1.0)
                    all_probs.append(prob)
                    per_question_probs[q_idx].append(prob)
                    text_probs.append(prob)

            per_text_probs[tid] = {
                "mean_confidence": round(float(np.mean(text_probs)), 6) if text_probs else None,
                "min_confidence": round(float(np.min(text_probs)), 6) if text_probs else None,
                "std_confidence": round(float(np.std(text_probs)), 6) if text_probs else None,
            }

        all_probs_arr = np.array(all_probs) if all_probs else np.array([1.0])

        per_question_summary = {}
        for q_idx, family in enumerate(question_families):
            q_probs = np.array(per_question_probs[q_idx]) if per_question_probs[q_idx] else np.array([1.0])
            per_question_summary[family] = {
                "mean_confidence": round(float(np.mean(q_probs)), 6),
                "min_confidence": round(float(np.min(q_probs)), 6),
                "std_confidence": round(float(np.std(q_probs)), 6),
                "pct_below_90": round(float(np.mean(q_probs < 0.90)) * 100, 2),
                "pct_below_50": round(float(np.mean(q_probs < 0.50)) * 100, 2),
            }

        summary[scale] = {
            "overall": {
                "mean_confidence": round(float(np.mean(all_probs_arr)), 6),
                "median_confidence": round(float(np.median(all_probs_arr)), 6),
                "min_confidence": round(float(np.min(all_probs_arr)), 6),
                "max_confidence": round(float(np.max(all_probs_arr)), 6),
                "std_confidence": round(float(np.std(all_probs_arr)), 6),
                "pct_below_90": round(float(np.mean(all_probs_arr < 0.90)) * 100, 2),
                "pct_below_50": round(float(np.mean(all_probs_arr < 0.50)) * 100, 2),
                "n_scores": len(all_probs),
            },
            "per_question": per_question_summary,
            "per_text": per_text_probs,
        }

    return summary


# ─── Entry Point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run batch experiment with logprobs")
    parser.add_argument("--samples", type=int, default=20, help="Samples per scale (default: 20)")
    args = parser.parse_args()

    asyncio.run(run_experiment(n_samples=args.samples))
