"""
FastAPI Backend for QA-Embedding Granularity Analyzer

Job-based async processing with cancellation support.

LOCAL DEV vs PRODUCTION:
- Set ALLOW_ENV_API_KEY=true for local dev (allows fallback to .env key)
- In production, don't set this - users must provide their own key
"""

import asyncio
import logging
import os
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from tqdm import tqdm

from .analysis import compute_all_metrics
from .client import QAEmbeddingClient
from .config import get_api_key_for_local_dev
from .questions import load_questions, VALID_MODES, DEFAULT_MODE, get_mode_description
from .scales import SCALE_ORDER

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("grading_llm")


def is_local_dev_mode() -> bool:
    """Check if running in local dev mode (allows env API key fallback)."""
    return os.environ.get("ALLOW_ENV_API_KEY", "").lower() == "true"

app = FastAPI(
    title="QA-Embedding Granularity API",
    description="Measure LLM consistency across grading granularities",
    version="0.1.0",
)

# CORS for frontend (localhost dev + Vercel production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class Job:
    """Represents an analysis job."""
    def __init__(self, job_id: str, statement: str, question_mode: str = DEFAULT_MODE):
        self.job_id = job_id
        self.statement = statement
        self.question_mode = question_mode
        self.status = JobStatus.PENDING
        self.progress = 0  # 0-100
        self.current_scale = ""
        self.current_sample = 0
        self.total_samples = 0
        self.result: Optional[Dict[str, Any]] = None
        self.error: Optional[str] = None
        self.cancelled = False
        self.task: Optional[asyncio.Task] = None
        self.created_at = datetime.now()
        self.usage: Optional[Dict[str, Any]] = None  # Token usage and cost


# In-memory job storage (cleared on restart)
jobs: Dict[str, Job] = {}


class ValidateKeyRequest(BaseModel):
    api_key: str


class ValidateKeyResponse(BaseModel):
    valid: bool
    error: Optional[str] = None


class StartJobRequest(BaseModel):
    statement: str
    api_key: Optional[str] = None  # Required in production, optional in local dev
    n_samples: int = 20
    question_mode: str = DEFAULT_MODE  # 'mech' or 'interp'


class StartJobResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int
    current_scale: str
    current_sample: int
    total_samples: int
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    usage: Optional[Dict[str, Any]] = None  # Token usage and cost


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.get("/api/config")
async def get_config():
    """
    Get app configuration.
    Returns whether API key is required (false in local dev mode with env key available).
    """
    if is_local_dev_mode():
        try:
            get_api_key_for_local_dev()
            return {"require_api_key": False, "mode": "local_dev"}
        except ValueError:
            return {"require_api_key": True, "mode": "local_dev_no_key"}
    return {"require_api_key": True, "mode": "production"}


@app.get("/api/questions")
async def get_questions(mode: str = DEFAULT_MODE):
    """
    Get the 20 questions for specified mode.

    Args:
        mode: Question mode ('mech' or 'interp')
    """
    if mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode '{mode}'. Must be one of: {VALID_MODES}"
        )
    try:
        questions = load_questions(mode=mode)
        return {
            "questions": questions,
            "mode": mode,
            "mode_description": get_mode_description(mode)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/question-modes")
async def get_question_modes():
    """Get available question modes and their descriptions."""
    return {
        "modes": VALID_MODES,
        "default": DEFAULT_MODE,
        "descriptions": {
            mode: get_mode_description(mode) for mode in VALID_MODES
        }
    }


@app.post("/api/validate-key", response_model=ValidateKeyResponse)
async def validate_api_key(request: ValidateKeyRequest):
    """
    Validate an OpenAI API key by making a minimal test call to GPT-4o mini.
    Returns valid=true if the key works, valid=false with error message otherwise.
    """
    from openai import AsyncOpenAI, AuthenticationError, APIError

    if not request.api_key or not request.api_key.strip():
        return ValidateKeyResponse(valid=False, error="API key is required")

    api_key = request.api_key.strip()

    # Basic format check
    if not api_key.startswith("sk-"):
        return ValidateKeyResponse(valid=False, error="Invalid API key format. Key should start with 'sk-'")

    try:
        client = AsyncOpenAI(api_key=api_key)

        # Make a minimal test call to GPT-4o mini
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=1,
        )

        # If we get here, the key is valid
        return ValidateKeyResponse(valid=True)

    except AuthenticationError:
        return ValidateKeyResponse(valid=False, error="Invalid API key. Please check your key and try again.")
    except APIError as e:
        return ValidateKeyResponse(valid=False, error=f"API error: {str(e)}")
    except Exception as e:
        return ValidateKeyResponse(valid=False, error=f"Validation failed: {str(e)}")


@app.post("/api/jobs/start", response_model=StartJobResponse)
async def start_job(request: StartJobRequest, background_tasks: BackgroundTasks):
    """Start a new analysis job. Returns job_id for polling."""

    # Validate question mode
    question_mode = request.question_mode
    if question_mode not in VALID_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid question_mode '{question_mode}'. Must be one of: {VALID_MODES}"
        )

    # Get API key - required in production, optional in local dev
    api_key = (request.api_key or "").strip()

    if not api_key:
        # Try local dev fallback
        if is_local_dev_mode():
            try:
                api_key = get_api_key_for_local_dev()
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail="No API key provided and none found in environment.",
                )
        else:
            raise HTTPException(
                status_code=400,
                detail="API key is required. Please provide your OpenAI API key.",
            )

    # Create job with question mode
    job_id = str(uuid.uuid4())[:8]
    job = Job(job_id, request.statement, question_mode=question_mode)
    job.total_samples = request.n_samples * len(SCALE_ORDER)
    jobs[job_id] = job

    # Log job start
    statement_preview = request.statement[:50] + "..." if len(request.statement) > 50 else request.statement
    logger.info(f"Job {job_id} STARTED | Mode: {question_mode} | Statement: \"{statement_preview}\"")

    # Start background task
    task = asyncio.create_task(
        run_analysis(job, api_key, request.n_samples)
    )
    job.task = task

    return StartJobResponse(job_id=job_id, status=job.status.value)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll job status. Returns progress and result when complete."""

    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    return JobStatusResponse(
        job_id=job.job_id,
        status=job.status.value,
        progress=job.progress,
        current_scale=job.current_scale,
        current_sample=job.current_sample,
        total_samples=job.total_samples,
        result=job.result,
        error=job.error,
        usage=job.usage,
    )


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a running job and clean up."""

    # Handle case where job doesn't exist (might be from sendBeacon after job completed)
    if job_id not in jobs:
        logger.info(f"Job {job_id} | Cancel request - job not found (already completed or cleaned up)")
        return {"status": "not_found"}

    job = jobs[job_id]

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]:
        logger.info(f"Job {job_id} | Cancel request - already {job.status.value}")
        return {"status": "already_finished", "job_status": job.status.value}

    logger.info(f"Job {job_id} | CANCELLING...")

    # Mark as cancelled
    job.cancelled = True
    job.status = JobStatus.CANCELLED

    # Cancel the task if running
    if job.task and not job.task.done():
        job.task.cancel()
        try:
            await job.task
        except asyncio.CancelledError:
            pass

    # Remove from jobs dict (forget the job)
    del jobs[job_id]

    logger.info(f"Job {job_id} | CANCELLED and cleaned up")
    return {"status": "cancelled"}


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a completed/failed job from memory."""

    if job_id not in jobs:
        logger.info(f"Job {job_id} | Delete request - job not found")
        return {"status": "not_found"}

    job = jobs[job_id]

    # Cancel if still running
    if job.task and not job.task.done():
        logger.info(f"Job {job_id} | Delete request - cancelling running task")
        job.cancelled = True
        job.task.cancel()
        try:
            await job.task
        except asyncio.CancelledError:
            pass

    del jobs[job_id]
    logger.info(f"Job {job_id} | DELETED from memory")
    return {"status": "deleted"}


async def run_analysis(job: Job, api_key: str, n_samples: int):
    """Run the analysis in background with progress updates."""

    start_time = time.time()

    try:
        job.status = JobStatus.RUNNING

        # Load questions for the specified mode
        questions = load_questions(mode=job.question_mode)

        # Initialize client
        client = QAEmbeddingClient(
            api_key=api_key,
            model="gpt-4o-mini",
            batch_size=20,
        )

        embeddings = {}
        completed_samples = 0

        # Create tqdm progress bar for terminal
        with tqdm(
            total=job.total_samples,
            desc=f"Job {job.job_id}",
            unit="sample",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        ) as pbar:

            for scale_idx, scale_name in enumerate(SCALE_ORDER):
                if job.cancelled:
                    logger.info(f"Job {job.job_id} CANCELLED")
                    return

                job.current_scale = scale_name
                logger.info(f"Job {job.job_id} | Scale: {scale_name} | Starting...")
                pbar.set_postfix(scale=scale_name)

                # Create a closure to capture scale_idx correctly
                def make_progress_callback(sidx, pbar_ref):
                    def callback(sample_num):
                        update_progress(job, sidx, sample_num, n_samples)
                        pbar_ref.update(1)
                        pbar_ref.set_postfix(scale=job.current_scale, sample=f"{sample_num}/{n_samples}")
                    return callback

                # Generate embeddings for this scale
                scale_embeddings = await client.query_all_questions_with_progress(
                    job.statement,
                    questions,
                    scale_name,
                    n_samples=n_samples,
                    on_sample_complete=make_progress_callback(scale_idx, pbar),
                    cancel_check=lambda: job.cancelled,
                )

                if job.cancelled:
                    logger.info(f"Job {job.job_id} CANCELLED")
                    return

                embeddings[scale_name] = scale_embeddings
                completed_samples += n_samples
                job.progress = int((completed_samples / job.total_samples) * 100)
                logger.info(f"Job {job.job_id} | Scale: {scale_name} | Complete ({job.progress}%)")

        if job.cancelled:
            logger.info(f"Job {job.job_id} CANCELLED")
            return

        # Compute metrics
        logger.info(f"Job {job.job_id} | Computing metrics...")
        metrics = compute_all_metrics(embeddings, questions)

        # Get usage stats
        usage = client.get_usage()
        job.usage = usage.to_dict()

        # Store result - include all metrics for frontend
        job.result = {
            "run_id": job.job_id,
            "statement": job.statement,
            "question_mode": job.question_mode,
            "question_mode_description": get_mode_description(job.question_mode),
            "scale_metrics": metrics["scale_metrics"],
            "pca": metrics["pca"],
            "aggregate": metrics["aggregate"],
            "identical_counts": metrics["identical_counts"],
            "embeddings_raw": metrics["embeddings_raw"],
        }
        job.status = JobStatus.COMPLETED
        job.progress = 100

        # Log completion
        duration = time.time() - start_time
        logger.info(
            f"Job {job.job_id} COMPLETED | "
            f"Duration: {duration:.1f}s | "
            f"Cost: ${usage.cost_usd:.4f} | "
            f"Tokens: {usage.total_tokens:,}"
        )

    except asyncio.CancelledError:
        job.status = JobStatus.CANCELLED
        logger.info(f"Job {job.job_id} CANCELLED (async)")
        raise
    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        logger.error(f"Job {job.job_id} FAILED | Error: {str(e)}")


def update_progress(job: Job, scale_idx: int, sample_num: int, samples_per_scale: int):
    """Update job progress."""
    if job.cancelled:
        return

    job.current_sample = sample_num
    completed = (scale_idx * samples_per_scale) + sample_num
    job.progress = int((completed / job.total_samples) * 100)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the FastAPI server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
