"""
Async OpenAI Client with Batching

Handles all API calls with:
- Batching by 20 questions per call
- Exponential backoff on rate limits
- JSON response validation with retry
- Concurrency limiting
- Token usage and cost tracking
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
from openai import AsyncOpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from tqdm import tqdm

# Logger for this module
logger = logging.getLogger("grading_llm.client")

from .config import get_api_key_for_local_dev
from .prompts import create_batched_prompt, create_single_question_prompt
from .scales import validate_response, SCALE_ORDER


# GPT-4o-mini pricing (as of 2024)
PRICING = {
    "gpt-4o-mini": {
        "input": 0.150 / 1_000_000,   # $0.150 per 1M input tokens
        "output": 0.600 / 1_000_000,  # $0.600 per 1M output tokens
    },
    "gpt-4o": {
        "input": 2.50 / 1_000_000,    # $2.50 per 1M input tokens
        "output": 10.00 / 1_000_000,  # $10.00 per 1M output tokens
    },
}


@dataclass
class UsageStats:
    """Tracks API usage and costs."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    cost_usd: float = 0.0

    def add(self, input_tokens: int, output_tokens: int, model: str):
        """Add usage from an API call."""
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.total_tokens += input_tokens + output_tokens
        self.api_calls += 1

        # Calculate cost
        if model in PRICING:
            self.cost_usd += (
                input_tokens * PRICING[model]["input"] +
                output_tokens * PRICING[model]["output"]
            )

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "cost_usd": round(self.cost_usd, 6),
        }


class QAEmbeddingClient:
    """
    Async client for generating QA embeddings.

    Batches questions in groups of 20 for efficiency.
    Tracks token usage and costs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        max_concurrent: int = 10,
        batch_size: int = 20,
        temperature: float = 0.0,
    ):
        """
        Initialize the client.

        Args:
            api_key: OpenAI API key. If None, uses get_api_key_for_local_dev()
            model: Model to use
            max_concurrent: Maximum concurrent API calls
            batch_size: Questions per API call
            temperature: Temperature for generation (0 for determinism)
        """
        if api_key is None:
            api_key = get_api_key_for_local_dev()

        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.temperature = temperature
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.usage = UsageStats()

    def reset_usage(self):
        """Reset usage statistics."""
        self.usage = UsageStats()

    def get_usage(self) -> UsageStats:
        """Get current usage statistics."""
        return self.usage

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=1, max=60),
        retry=retry_if_exception_type((Exception,)),
    )
    async def _call_api(self, prompt: str, max_tokens: int = 500) -> str:
        """Make a single API call with retry logic."""
        async with self.semaphore:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens,
            )

            # Track usage
            if response.usage:
                self.usage.add(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    self.model
                )

            return response.choices[0].message.content.strip()

    def _parse_single_response(self, content: str, scale_name: str) -> float:
        """
        Parse a single-question response.

        Args:
            content: Raw response content
            scale_name: Scale for validation

        Returns:
            Validated score
        """
        # Try JSON parse
        try:
            data = json.loads(content)
            if "score" in data:
                return validate_response(data["score"], scale_name)
        except json.JSONDecodeError:
            pass

        # Fallback: regex for score
        match = re.search(r'"score"\s*:\s*([\d.]+)', content)
        if match:
            return validate_response(float(match.group(1)), scale_name)

        # Last resort: find any float
        match = re.search(r'(\d+\.?\d*)', content)
        if match:
            return validate_response(float(match.group(1)), scale_name)

        # Default to 0
        return 0.0

    def _parse_batched_response(
        self, content: str, n_questions: int, scale_name: str
    ) -> List[float]:
        """
        Parse a batched response.

        Args:
            content: Raw response content
            n_questions: Expected number of scores
            scale_name: Scale for validation

        Returns:
            List of validated scores
        """
        scores = []

        # Try JSON parse
        try:
            data = json.loads(content)
            if "scores" in data and isinstance(data["scores"], list):
                for s in data["scores"]:
                    scores.append(validate_response(s, scale_name))
        except json.JSONDecodeError:
            pass

        # Fallback: regex for array
        if not scores:
            match = re.search(r'"scores"\s*:\s*\[([\d.,\s]+)\]', content)
            if match:
                nums = re.findall(r'[\d.]+', match.group(1))
                for n in nums:
                    scores.append(validate_response(float(n), scale_name))

        # Fallback: find all numbers
        if not scores:
            nums = re.findall(r'(?<!["\w])(\d+\.?\d*)(?!["\w])', content)
            for n in nums[:n_questions]:
                scores.append(validate_response(float(n), scale_name))

        # Pad with zeros if needed
        while len(scores) < n_questions:
            scores.append(0.0)

        return scores[:n_questions]

    async def query_single(
        self, statement: str, question: str, scale_name: str
    ) -> Tuple[float, Dict]:
        """
        Query a single question.

        Args:
            statement: Statement to evaluate
            question: Question to answer
            scale_name: Grading scale

        Returns:
            Tuple of (score, raw_response_dict)
        """
        prompt = create_single_question_prompt(statement, question, scale_name)
        content = await self._call_api(prompt)
        score = self._parse_single_response(content, scale_name)

        return score, {
            "statement": statement,
            "question": question,
            "scale": scale_name,
            "raw_response": content,
            "score": score,
        }

    async def query_batch(
        self, statement: str, questions: List[Dict], scale_name: str,
        batch_num: int = 0, total_batches: int = 0
    ) -> Tuple[List[float], Dict]:
        """
        Query a batch of questions (up to batch_size).

        Args:
            statement: Statement to evaluate
            questions: List of question dicts
            scale_name: Grading scale
            batch_num: Current batch number (for logging)
            total_batches: Total number of batches (for logging)

        Returns:
            Tuple of (scores_list, raw_response_dict)
        """
        prompt = create_batched_prompt(
            statement, questions, scale_name, self.batch_size
        )

        # Track tokens before call
        tokens_before = self.usage.total_tokens

        content = await self._call_api(prompt, max_tokens=1000)
        scores = self._parse_batched_response(content, len(questions), scale_name)

        # Log batch completion with token count
        tokens_used = self.usage.total_tokens - tokens_before
        if batch_num > 0 and total_batches > 0:
            logger.info(f"      Batch {batch_num}/{total_batches} complete ({tokens_used} tokens)")

        return scores, {
            "statement": statement,
            "questions": [q.get("id", str(i)) for i, q in enumerate(questions)],
            "scale": scale_name,
            "raw_response": content,
            "scores": scores,
        }

    async def query_all_questions(
        self,
        statement: str,
        questions: List[Dict],
        scale_name: str,
        n_samples: int = 20,
        on_response: Optional[callable] = None,
    ) -> np.ndarray:
        """
        Query all questions multiple times for one scale.

        Args:
            statement: Statement to evaluate
            questions: List of question dicts
            scale_name: Grading scale
            n_samples: Number of times to repeat
            on_response: Optional callback for each response

        Returns:
            Array of shape (n_samples, n_questions)
        """
        n_questions = len(questions)
        results = np.zeros((n_samples, n_questions))

        for sample_idx in range(n_samples):
            # Process in batches
            all_scores = []
            for batch_start in range(0, n_questions, self.batch_size):
                batch_end = min(batch_start + self.batch_size, n_questions)
                batch_questions = questions[batch_start:batch_end]

                scores, response = await self.query_batch(
                    statement, batch_questions, scale_name
                )
                all_scores.extend(scores)

                if on_response:
                    on_response(response)

            results[sample_idx] = all_scores

        return results

    async def generate_full_embedding(
        self,
        statement: str,
        questions: List[Dict],
        n_samples: int = 20,
        on_response: Optional[callable] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all scales.

        Args:
            statement: Statement to evaluate
            questions: List of question dicts
            n_samples: Number of samples per scale
            on_response: Optional callback for each response

        Returns:
            Dict mapping scale_name -> array of shape (n_samples, n_questions)
        """
        embeddings = {}

        for scale_name in SCALE_ORDER:
            embeddings[scale_name] = await self.query_all_questions(
                statement, questions, scale_name, n_samples, on_response
            )

        return embeddings

    async def query_all_questions_with_progress(
        self,
        statement: str,
        questions: List[Dict],
        scale_name: str,
        n_samples: int = 20,
        on_sample_complete: Optional[Callable[[int], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> np.ndarray:
        """
        Query all questions with progress callbacks and cancellation support.

        Args:
            statement: Statement to evaluate
            questions: List of question dicts
            scale_name: Grading scale
            n_samples: Number of times to repeat
            on_sample_complete: Callback when each sample completes (sample_num)
            cancel_check: Function that returns True if cancelled

        Returns:
            Array of shape (n_samples, n_questions)
        """
        n_questions = len(questions)
        results = np.zeros((n_samples, n_questions))

        # Calculate total batches per sample
        total_batches = (n_questions + self.batch_size - 1) // self.batch_size

        for sample_idx in range(n_samples):
            # Check for cancellation
            if cancel_check and cancel_check():
                return results[:sample_idx] if sample_idx > 0 else np.array([])

            logger.info(f"    Sample {sample_idx + 1}/{n_samples} starting...")

            # Process in batches
            all_scores = []
            batch_num = 0
            for batch_start in range(0, n_questions, self.batch_size):
                # Check for cancellation before each API call
                if cancel_check and cancel_check():
                    return results[:sample_idx] if sample_idx > 0 else np.array([])

                batch_num += 1
                batch_end = min(batch_start + self.batch_size, n_questions)
                batch_questions = questions[batch_start:batch_end]

                scores, _ = await self.query_batch(
                    statement, batch_questions, scale_name,
                    batch_num=batch_num, total_batches=total_batches
                )
                all_scores.extend(scores)

            results[sample_idx] = all_scores

            # Log sample completion
            logger.info(f"    Sample {sample_idx + 1}/{n_samples} complete")

            # Notify progress
            if on_sample_complete:
                on_sample_complete(sample_idx + 1)

        return results

    async def generate_full_embedding_with_tqdm(
        self,
        statement: str,
        questions: List[Dict],
        n_samples: int = 20,
        show_progress: bool = True,
    ) -> Dict[str, np.ndarray]:
        """
        Generate embeddings for all scales with tqdm progress bar.

        Args:
            statement: Statement to evaluate
            questions: List of question dicts
            n_samples: Number of samples per scale
            show_progress: Whether to show tqdm progress bar

        Returns:
            Dict mapping scale_name -> array of shape (n_samples, n_questions)
        """
        embeddings = {}
        total_calls = len(SCALE_ORDER) * n_samples

        with tqdm(
            total=total_calls,
            desc="Processing",
            disable=not show_progress,
            unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        ) as pbar:
            for scale_name in SCALE_ORDER:
                pbar.set_description(f"Scale: {scale_name}")
                n_questions = len(questions)
                results = np.zeros((n_samples, n_questions))

                for sample_idx in range(n_samples):
                    all_scores = []
                    for batch_start in range(0, n_questions, self.batch_size):
                        batch_end = min(batch_start + self.batch_size, n_questions)
                        batch_questions = questions[batch_start:batch_end]

                        scores, _ = await self.query_batch(
                            statement, batch_questions, scale_name
                        )
                        all_scores.extend(scores)

                    results[sample_idx] = all_scores
                    pbar.update(1)

                embeddings[scale_name] = results

        return embeddings
