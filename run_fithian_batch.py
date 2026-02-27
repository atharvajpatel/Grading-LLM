"""
Batch runner for Fithian analysis.
Runs 10 texts through the full Grading-LLM pipeline.
Saves raw results as JSON for notebook analysis.
"""

import asyncio
import json
import os
import time
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from src.grading_llm.client import QAEmbeddingClient
from src.grading_llm.questions import load_questions
from src.grading_llm.scales import SCALE_ORDER
from src.grading_llm.analysis import compute_all_metrics

API_KEY = os.getenv("OPENAI_API_KEY")
assert API_KEY, "Set OPENAI_API_KEY in .env"

SAMPLES = 20
MODE = "mech"
INPUT_FILE = "data/batch_input_10.jsonl"
OUTPUT_FILE = "data/batch_results.json"


def serializeValue(obj):
    """JSON serializer for numpy types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


async def runSingleText(textId: str, text: str, questions: list) -> dict:
    """Run full analysis on one text, return raw data."""
    client = QAEmbeddingClient(
        api_key=API_KEY,
        model="gpt-4o-mini",
        batch_size=20,
    )

    # generate_full_embedding_with_tqdm handles all 4 scales
    # Returns Dict[scale_name -> np.ndarray(n_samples, n_questions)]
    embeddings = await client.generate_full_embedding_with_tqdm(
        statement=text,
        questions=questions,
        n_samples=SAMPLES,
        show_progress=True,
    )

    # Compute metrics using the existing analysis module
    metrics = compute_all_metrics(embeddings, questions)

    # Get usage stats
    usage = client.get_usage()

    # Convert numpy arrays to lists for JSON serialization
    rawScores = {}
    for scaleName, arr in embeddings.items():
        rawScores[scaleName] = arr.tolist()

    return {
        "id": textId,
        "text": text,
        "raw_scores": rawScores,
        "metrics": json.loads(json.dumps(metrics, default=serializeValue)),
        "usage": usage.to_dict(),
    }


async def main():
    """Run batch analysis on all 10 texts."""
    # Load questions
    questions = load_questions(mode=MODE)
    print(f"Loaded {len(questions)} questions (mode: {MODE})")

    # Load input texts
    texts = []
    with open(INPUT_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(json.loads(line))

    totalCalls = len(texts) * SAMPLES * len(SCALE_ORDER)
    print(f"Running batch analysis on {len(texts)} texts...")
    print(f"Total samples: {totalCalls} ({len(texts)} texts x {SAMPLES} samples x {len(SCALE_ORDER)} scales)")
    print(f"Estimated API calls: ~{len(texts) * SAMPLES * len(SCALE_ORDER)} (batched 20 questions per call)")
    print(f"Estimated time: 10-20 minutes\n")

    allResults = []
    totalStartTime = time.time()

    for i, item in enumerate(texts):
        textId = item.get("id", f"text_{i}")
        text = item["text"]
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"[{i+1}/{len(texts)}] Processing: {textId}")
        print(f"  Text: {preview}")

        startTime = time.time()
        result = await runSingleText(textId, text, questions)
        duration = time.time() - startTime

        allResults.append(result)
        cost = result["usage"]["cost_usd"]
        print(f"  Done in {duration:.1f}s | Cost: ${cost:.4f} | Scales: {list(result['raw_scores'].keys())}\n")

    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump(allResults, f, indent=2, default=serializeValue)

    totalDuration = time.time() - totalStartTime
    totalCost = sum(r["usage"]["cost_usd"] for r in allResults)
    print(f"All results saved to {OUTPUT_FILE}")
    print(f"Total texts: {len(allResults)}")
    print(f"Total time: {totalDuration:.1f}s ({totalDuration/60:.1f} min)")
    print(f"Total cost: ${totalCost:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
