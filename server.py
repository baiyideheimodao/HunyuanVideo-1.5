#!/usr/bin/env python3
"""HunyuanVideo-1.5 persistent inference server.

Provides REST API endpoints for text-to-video and image-to-video generation
with progress tracking and GPU monitoring. The pipeline is loaded once at
startup and kept resident in memory across requests.

Usage::

    python server.py --model_path ./ckpts --resolution 480p --video_length 81
"""

from __future__ import annotations

import argparse
import copy
import io
import os
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

import einops
import imageio
import loguru
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from PIL import Image
from pydantic import BaseModel, Field

# ── Distributed setup (must happen before any CUDA operations) ──────────────

os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "29500")

from hyvideo.commons.parallel_states import initialize_parallel_state

initialize_parallel_state(sp=1)
torch.cuda.set_device(0)

from hyvideo.commons.infer_state import InferState
from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline


# ── Helper: save video to file ──────────────────────────────────────────────

def save_video_to_path(video: torch.Tensor, path: Union[str, Path]) -> None:
    """Save a video tensor to an mp4 file.

    Args:
        video: Tensor of shape (C, F, H, W) or (B, C, F, H, W), values in [0, 1].
        path: Output file path.
    """
    if video.ndim == 5:
        assert video.shape[0] == 1, f"Expected batch size 1, got {video.shape[0]}"
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, "c f h w -> f h w c")
    imageio.mimwrite(str(path), vid, fps=24)


# ── Global state ────────────────────────────────────────────────────────────

pipe: Optional[HunyuanVideo_1_5_Pipeline] = None
server_config: Optional[argparse.Namespace] = None
startup_time: Optional[float] = None

tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()
_inference_busy = threading.Event()
_inference_executor = ThreadPoolExecutor(max_workers=1)

# ── Pydantic models ─────────────────────────────────────────────────────────


class T2VRequest(BaseModel):
    """Request body for text-to-video generation."""

    prompt: str = Field(..., description="Text prompt describing the desired video", min_length=1)
    video_length: Optional[int] = Field(None, description="Override frame count (uses server default if not set)", ge=1, le=121)


class TaskResponse(BaseModel):
    """Response returned when a generation task is submitted or queried."""

    task_id: str
    status: str  # queued | running | completed | error
    step: int = 0
    total: int = 0
    output_url: Optional[str] = Field(None, description="Relative URL to download the video when completed")
    error: Optional[str] = Field(None, description="Error message if status is 'error'")


class GPUInfo(BaseModel):
    """GPU utilization and memory information."""

    gpu_name: str
    vram_total_mb: int
    vram_used_mb: int
    vram_free_mb: int
    utilization_pct: int


class HealthResponse(BaseModel):
    """Server health status."""

    status: str  # loading | ready | busy
    uptime_seconds: float
    gpu_info: GPUInfo


# ── Progress tracker (replaces tqdm during inference) ───────────────────────


class _ProgressTracker:
    """A tqdm-compatible progress tracker that writes to the tasks dict.

    Injected via ``pipe.progress_bar`` so the pipeline's per-step ``update()``
    calls update the task state visible to ``GET /task/{task_id}``.
    """

    def __init__(self, task_id: str, total: int) -> None:
        self.task_id = task_id
        self.total = total
        self.n = 0

    def update(self, n: int = 1) -> None:
        self.n += n
        with _tasks_lock:
            if self.task_id in _tasks:
                tasks[self.task_id]["step"] = self.n

    def __enter__(self) -> _ProgressTracker:
        return self

    def __exit__(self, *args: object) -> None:
        pass

    def __iter__(self):
        return self


# ── GPU monitoring ──────────────────────────────────────────────────────────


def _query_gpu() -> GPUInfo:
    """Query GPU status via nvidia-smi subprocess."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        parts = [x.strip() for x in result.stdout.strip().split(",")]
        return GPUInfo(
            gpu_name=parts[0],
            vram_total_mb=int(parts[1]),
            vram_used_mb=int(parts[2]),
            vram_free_mb=int(parts[3]),
            utilization_pct=int(parts[4]),
        )
    except Exception:
        loguru.logger.warning("Failed to query nvidia-smi")
        return GPUInfo(
            gpu_name="unknown",
            vram_total_mb=0,
            vram_used_mb=0,
            vram_free_mb=0,
            utilization_pct=0,
        )


# ── Inference runner (executed in background thread) ────────────────────────


def _run_inference(
    task_id: str,
    prompt: str,
    reference_image: Optional[Image.Image] = None,
    video_length: Optional[int] = None,
) -> None:
    """Run the video generation pipeline in a background thread.

    Updates *tasks[task_id]* throughout so the polling endpoint can report
    progress.  Serialised by ``_inference_executor`` (max_workers=1).
    """
    vl = video_length if video_length is not None else server_config.video_length
    _inference_busy.set()
    original_progress_bar = pipe.progress_bar

    try:
        with _tasks_lock:
            tasks[task_id]["status"] = "running"

        # ── Inject progress tracker ─────────────────────────────────────
        total_steps = server_config.num_inference_steps or 50
        tracker = _ProgressTracker(task_id, total_steps)

        def _custom_progress_bar(iterable=None, total=None):
            if total is not None:
                tracker.total = total
                return tracker
            if iterable is not None:
                return iterable
            return tracker

        pipe.progress_bar = _custom_progress_bar

        # ── Build kwargs ────────────────────────────────────────────────
        extra_kwargs: dict = {}
        if reference_image is not None:
            extra_kwargs["reference_image"] = reference_image

        # ── Run pipeline ────────────────────────────────────────────────
        out = pipe(
            prompt=prompt,
            aspect_ratio=server_config.aspect_ratio,
            video_length=vl,
            num_inference_steps=server_config.num_inference_steps,
            enable_sr=server_config.enable_sr,
            negative_prompt=server_config.negative_prompt,
            seed=server_config.seed,
            output_type="pt",
            prompt_rewrite=False,
            return_pre_sr_video=False,
            **extra_kwargs,
        )

        # ── Save output ─────────────────────────────────────────────────
        output_dir = Path(server_config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_filename = f"{task_id}.mp4"
        output_path = output_dir / output_filename

        videos = (
            out.sr_videos
            if server_config.enable_sr and getattr(out, "sr_videos", None) is not None
            else out.videos
        )
        save_video_to_path(videos, output_path)

        with _tasks_lock:
            tasks[task_id]["status"] = "completed"
            tasks[task_id]["output_url"] = f"/outputs/{output_filename}"

        loguru.logger.info(f"[{task_id}] completed → {output_path}")

    except Exception as exc:
        loguru.logger.error(f"[{task_id}] failed: {exc}")
        with _tasks_lock:
            tasks[task_id]["status"] = "error"
            tasks[task_id]["error"] = str(exc)
    finally:
        pipe.progress_bar = original_progress_bar
        _inference_busy.clear()


# ── FastAPI application ─────────────────────────────────────────────────────

app = FastAPI(
    title="HunyuanVideo-1.5 Inference Server",
    description=(
        "Persistent video generation service. "
        "The pipeline is loaded once at startup and reused across requests. "
        "Use `POST /generate/t2v` or `/generate/i2v` to submit a task, "
        "then poll `GET /task/{task_id}` for progress."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.on_event("startup")
async def _on_startup() -> None:
    """Load the pipeline on first start.  Models stay resident afterwards."""
    global pipe, startup_time

    loguru.logger.info("Loading pipeline — this may take 1–2 minutes …")
    t0 = time.time()

    # --- Resolve transformer version ---
    task = "t2v"
    cfg_distilled = getattr(server_config, "cfg_distilled", False)
    step_distilled = getattr(server_config, "enable_step_distill", False)
    sparse_attn = getattr(server_config, "sparse_attn", False)
    transformer_version = HunyuanVideo_1_5_Pipeline.get_transformer_version(
        server_config.resolution, task, cfg_distilled, step_distilled, sparse_attn
    )

    # --- Dtype ---
    dtype_map = {"bf16": torch.bfloat16, "fp32": torch.float32}
    transformer_dtype = dtype_map[server_config.dtype]

    # --- Offloading ---
    enable_offloading = server_config.offloading
    if server_config.group_offloading is None:
        offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
        enable_group_offloading = offloading_config["enable_group_offloading"]
    else:
        enable_group_offloading = (server_config.group_offloading == "true")
    overlap_group_offloading = getattr(server_config, "overlap_group_offloading", True)

    device = torch.device("cpu") if enable_offloading else torch.device("cuda")
    transformer_init_device = torch.device("cpu") if enable_group_offloading else device

    # --- Create pipeline ---
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=server_config.model_path,
        transformer_version=transformer_version,
        create_sr_pipeline=server_config.enable_sr,
        transformer_dtype=transformer_dtype,
        device=device,
        transformer_init_device=transformer_init_device,
    )

    # --- Apply optimisations ---
    infer_state = InferState()
    infer_state.total_steps = server_config.num_inference_steps or 50

    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=enable_offloading,
        enable_group_offloading=enable_group_offloading,
        overlap_group_offloading=overlap_group_offloading,
    )

    if server_config.enable_sr and hasattr(pipe, "sr_pipeline"):
        sr_infer_state = copy.deepcopy(infer_state)
        sr_infer_state.enable_cache = False
        pipe.sr_pipeline.apply_infer_optimization(
            infer_state=sr_infer_state,
            enable_offloading=enable_offloading,
            enable_group_offloading=False,
        )

    startup_time = time.time()
    loguru.logger.info(f"Pipeline ready in {startup_time - t0:.1f}s")


# ── API endpoints ───────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def _root() -> RedirectResponse:
    """Redirect root to Swagger docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Server health check — returns loading/ready/busy status and GPU info."""
    gpu = _query_gpu()

    if pipe is None:
        status = "loading"
    elif _inference_busy.is_set():
        status = "busy"
    else:
        status = "ready"

    uptime = time.time() - startup_time if startup_time else 0.0
    return HealthResponse(status=status, uptime_seconds=uptime, gpu_info=gpu)


@app.get("/gpu", response_model=GPUInfo)
async def gpu_info() -> GPUInfo:
    """Current GPU utilisation and VRAM usage."""
    return _query_gpu()


@app.post("/generate/t2v", response_model=TaskResponse, status_code=202)
async def generate_t2v(req: T2VRequest) -> TaskResponse:
    """Submit a **text-to-video** generation task.

    Returns a ``task_id`` immediately (HTTP 202).
    Poll ``GET /task/{task_id}`` to track progress and retrieve the result.
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    task_id = str(uuid.uuid4())[:8]
    total_steps = server_config.num_inference_steps or 50

    with _tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "step": 0,
            "total": total_steps,
            "output_url": None,
            "error": None,
        }

    _inference_executor.submit(_run_inference, task_id, req.prompt, None, req.video_length)
    loguru.logger.info(f"[{task_id}] queued — t2v prompt={req.prompt[:60]}…")

    return TaskResponse(**tasks[task_id])


@app.post("/generate/i2v", response_model=TaskResponse, status_code=202)
async def generate_i2v(
    image: UploadFile = File(..., description="Reference image (JPEG/PNG)"),
    prompt: str = Form(..., description="Text prompt describing the desired motion"),
    video_length: Optional[int] = Form(None, description="Override frame count (uses server default if not set)", ge=1, le=121),
) -> TaskResponse:
    """Submit an **image-to-video** generation task.

    Upload a reference image and a text prompt.  Returns a ``task_id`` immediately.
    Poll ``GET /task/{task_id}`` for progress.
    """
    if pipe is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded yet")

    # Read uploaded image
    image_bytes = await image.read()
    reference_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    task_id = str(uuid.uuid4())[:8]
    total_steps = server_config.num_inference_steps or 50

    with _tasks_lock:
        tasks[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "step": 0,
            "total": total_steps,
            "output_url": None,
            "error": None,
        }

    _inference_executor.submit(_run_inference, task_id, prompt, reference_image, video_length)

    return TaskResponse(**tasks[task_id])


@app.get("/task/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str) -> TaskResponse:
    """Query the status and progress of a generation task.

    Returns ``step`` / ``total`` while running, and ``output_url`` when complete.
    """
    with _tasks_lock:
        if task_id not in tasks:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return TaskResponse(**tasks[task_id])


@app.get("/outputs/{filename:path}")
async def download_output(filename: str) -> FileResponse:
    """Download a generated video file."""
    filepath = Path(server_config.output_dir) / filename
    if not filepath.is_file():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found")
    return FileResponse(str(filepath), media_type="video/mp4")


# ── CLI entry point ─────────────────────────────────────────────────────────


def _parse_bool(value: str) -> bool:
    """Accept common boolean representations."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean: {value}")


def main() -> None:
    """Parse CLI arguments and start the uvicorn server."""
    parser = argparse.ArgumentParser(
        description="HunyuanVideo-1.5 persistent inference server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python server.py --model_path ./ckpts --resolution 480p\n"
            "  python server.py --model_path ./ckpts --resolution 720p --video_length 81 --port 8080\n"
        ),
    )

    # --- Required ---
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path to pretrained model checkpoint directory",
    )

    # --- Inference defaults (fixed for all API requests) ---
    parser.add_argument(
        "--resolution", type=str, default="480p", choices=["480p", "720p"],
        help="Video resolution (default: 480p)",
    )
    parser.add_argument(
        "--video_length", type=int, default=121,
        help="Number of frames per video (default: 121)",
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=None,
        help="Denoising steps; uses pipeline default if not set",
    )
    parser.add_argument(
        "--aspect_ratio", type=str, default="16:9",
        help="Video aspect ratio (default: 16:9)",
    )
    parser.add_argument(
        "--enable_sr", type=_parse_bool, default=True,
        help="Enable super-resolution pass (default: True)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed; random if not set",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp32"],
        help="Model precision (default: bf16)",
    )
    parser.add_argument(
        "--negative_prompt", type=str, default="",
        help="Default negative prompt applied to all requests",
    )

    # --- Offloading ---
    parser.add_argument(
        "--offloading", type=_parse_bool, default=True,
        help="Enable CPU offloading (default: True)",
    )
    parser.add_argument(
        "--group_offloading", type=str, default=None, choices=["true", "false"],
        help="Enable group offloading (true/false); auto-detect if not set",
    )
    parser.add_argument(
        "--overlap_group_offloading", type=_parse_bool, default=True,
        help="Overlap group offloading transfers (default: True)",
    )

    # --- Server ---
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="Listen address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8000,
        help="Listen port (default: 8000)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./outputs",
        help="Directory for generated video files (default: ./outputs)",
    )

    global server_config
    server_config = parser.parse_args()

    # Ensure output directory exists
    Path(server_config.output_dir).mkdir(parents=True, exist_ok=True)

    loguru.logger.info(f"Starting server — {vars(server_config)}")
    uvicorn.run(app, host=server_config.host, port=server_config.port)


if __name__ == "__main__":
    main()
