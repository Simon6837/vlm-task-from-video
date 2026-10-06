"""Describe the workflows in every MP4 under a directory with Ollama Vision.

Llama 3.2 Vision accepts images rather than video files. This script therefore
combines evenly sampled video frames into one contact-sheet image and attaches
it to one chat request for each video.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "OpenCV is required. Install dependencies with: python -m pip install -r requirements.txt"
    ) from exc

try:
    import ollama
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit(
        "The Ollama Python library is required. Install dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = SCRIPT_DIR / "videos"
RESULTS_DIR = SCRIPT_DIR / "results"
DEFAULT_FPS = 1
MIN_CONTEXT_WINDOW = 4096
IMAGE_TOKEN_ESTIMATE = 256
RESPONSE_TOKEN_RESERVE = 2048
CONTEXT_WINDOW_ALIGNMENT = 4096
CONTACT_SHEET_COLUMNS = 5
CONTACT_SHEET_TILE_WIDTH = 320
CONTACT_SHEET_TILE_HEIGHT = 180
CONTACT_SHEET_LABEL_HEIGHT = 24
PROMPT = (
    "The supplied image is a timestamped contact sheet made from a video. "
    "The person in the video is performing a task. Return only a JSON array "
    "with one object for each distinct action in chronological order. Each object "
    "must contain these keys: action, object, target, and motion."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract video frames and ask Ollama Vision for each workflow."
    )
    parser.add_argument(
        "model",
        help="Ollama model to pull and use, for example: gemma4:e4b",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory to scan recursively for MP4 files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Frames to sample per second of video (default: {DEFAULT_FPS})",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        help=(
            "Ollama context window in tokens for each request "
            "(default: calculate from the extracted frames)"
        ),
    )
    return parser.parse_args()


def find_videos(input_dir: Path) -> list[Path]:
    return sorted(
        (path for path in input_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".mp4"),
        key=lambda path: path.as_posix().lower(),
    )


def output_path_for_model(model: str) -> Path:
    """Create a Windows-safe result filename while preserving the model tag."""
    safe_model = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "-", model).rstrip(". ")
    if not safe_model:
        safe_model = "model"
    return RESULTS_DIR / f"{safe_model}.csv"


def frame_indices_at_fps(total_frames: int, source_fps: float, target_fps: float) -> list[int]:
    """Return unique source-frame indexes sampled from timestamp zero at target_fps."""
    if total_frames <= 0:
        raise ValueError("video reports no frames")
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("video reports no usable FPS")
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("requested FPS must be greater than zero")
    if target_fps >= source_fps:
        return list(range(total_frames))

    sample_count = math.ceil(total_frames * target_fps / source_fps)
    indices: list[int] = []
    for sample_number in range(sample_count):
        frame_index = int(sample_number * source_fps / target_fps)
        if frame_index >= total_frames:
            break
        if not indices or frame_index != indices[-1]:
            indices.append(frame_index)
    return indices


def round_up_to_context_window(value: int) -> int:
    """Round a token requirement up to an Ollama-friendly context size."""
    return max(
        MIN_CONTEXT_WINDOW,
        math.ceil(value / CONTEXT_WINDOW_ALIGNMENT) * CONTEXT_WINDOW_ALIGNMENT,
    )


def calculated_context_window(image_count: int) -> int:
    """Reserve image and response tokens for one video request."""
    if image_count < 1:
        raise ValueError("at least one image is required")
    return round_up_to_context_window(image_count * IMAGE_TOKEN_ESTIMATE + RESPONSE_TOKEN_RESERVE)


def context_window_from_error(error: Exception) -> int | None:
    """Return a safe context size when Ollama reports the actual prompt token count."""
    match = re.search(r'"n_prompt_tokens"\s*:\s*(\d+)', str(error))
    if match is None:
        return None
    return round_up_to_context_window(int(match.group(1)) + RESPONSE_TOKEN_RESERVE)


def timestamp_label(seconds: float) -> str:
    """Format a sampled frame's offset in a compact, human-readable form."""
    whole_seconds = max(0, int(seconds))
    minutes, seconds = divmod(whole_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def contact_sheet_canvas(frame_count: int) -> np.ndarray:
    """Create a white five-column contact-sheet canvas for sampled frames."""
    if frame_count < 1:
        raise ValueError("at least one extracted frame is required")
    rows = math.ceil(frame_count / CONTACT_SHEET_COLUMNS)
    tile_height = CONTACT_SHEET_LABEL_HEIGHT + CONTACT_SHEET_TILE_HEIGHT
    return np.full(
        (rows * tile_height, CONTACT_SHEET_COLUMNS * CONTACT_SHEET_TILE_WIDTH, 3),
        255,
        dtype=np.uint8,
    )


def add_frame_to_contact_sheet(
    sheet: np.ndarray, frame: np.ndarray, output_index: int, timestamp_seconds: float
) -> None:
    """Letterbox one frame in its contact-sheet tile and label its timestamp."""
    row, column = divmod(output_index, CONTACT_SHEET_COLUMNS)
    tile_x = column * CONTACT_SHEET_TILE_WIDTH
    tile_y = row * (CONTACT_SHEET_LABEL_HEIGHT + CONTACT_SHEET_TILE_HEIGHT)
    frame_height, frame_width = frame.shape[:2]
    scale = min(CONTACT_SHEET_TILE_WIDTH / frame_width, CONTACT_SHEET_TILE_HEIGHT / frame_height)
    resized_width = max(1, round(frame_width * scale))
    resized_height = max(1, round(frame_height * scale))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    frame_x = tile_x + (CONTACT_SHEET_TILE_WIDTH - resized_width) // 2
    frame_y = tile_y + CONTACT_SHEET_LABEL_HEIGHT + (CONTACT_SHEET_TILE_HEIGHT - resized_height) // 2
    sheet[frame_y : frame_y + resized_height, frame_x : frame_x + resized_width] = resized
    cv2.putText(
        sheet,
        timestamp_label(timestamp_seconds),
        (tile_x + 6, tile_y + 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


def create_contact_sheet(video_path: Path, destination: Path, fps: float) -> Path:
    """Create one timestamped contact-sheet JPEG from every sampled video frame."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("could not open video")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        frame_indices = frame_indices_at_fps(total_frames, source_fps, fps)
        sheet = contact_sheet_canvas(len(frame_indices))
        for output_index, frame_index in enumerate(frame_indices):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"could not decode frame {frame_index}")
            add_frame_to_contact_sheet(sheet, frame, output_index, frame_index / source_fps)

        sheet_path = destination / "contact_sheet.jpg"
        if not cv2.imwrite(str(sheet_path), sheet):
            raise RuntimeError("could not write temporary contact sheet")
        return sheet_path
    finally:
        capture.release()


def response_text(response: Any) -> str:
    """Extract text from both current and older ollama-python response shapes."""
    if isinstance(response, dict):
        return str(response["message"]["content"])
    return str(response.message.content)


class PromptExecutionError(Exception):
    """Keep the Ollama request duration when the request itself fails."""

    def __init__(self, cause: Exception, elapsed_seconds: float) -> None:
        self.cause = cause
        self.elapsed_seconds = elapsed_seconds
        super().__init__(str(cause))


def analyze_video(
    video_path: Path, model: str, fps: float, context_window: int | None
) -> tuple[str, float]:
    try:
        with tempfile.TemporaryDirectory(prefix="ollama_contact_sheet_") as temporary_directory:
            contact_sheet_path = create_contact_sheet(video_path, Path(temporary_directory), fps)
            selected_context_window = context_window or calculated_context_window(1)
            chat_arguments: dict[str, Any] = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT,
                        "images": [str(contact_sheet_path)],
                    }
                ],
                "options": {"num_ctx": selected_context_window},
            }
            started_at = time.perf_counter()
            try:
                response = ollama.chat(**chat_arguments)
            except Exception as exc:
                retry_context_window = (
                    context_window_from_error(exc) if context_window is None else None
                )
                if retry_context_window is None or retry_context_window <= selected_context_window:
                    raise PromptExecutionError(exc, time.perf_counter() - started_at) from exc
                chat_arguments["options"] = {"num_ctx": retry_context_window}
                try:
                    response = ollama.chat(**chat_arguments)
                except Exception as retry_exc:
                    raise PromptExecutionError(
                        retry_exc, time.perf_counter() - started_at
                    ) from retry_exc
            elapsed_seconds = time.perf_counter() - started_at
        return response_text(response).strip(), elapsed_seconds
    finally:
        unload_model(model)


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=["video_filename", "prompt_execution_seconds", "workflow"],
        )
        writer.writeheader()
        writer.writerows(rows)


def is_unsupported_mllama_error(error: Exception) -> bool:
    """Identify Ollama versions that cannot run Llama 3.2 Vision at all."""
    return "unknown model architecture: 'mllama'" in str(error).lower()


def pull_model(model: str) -> None:
    """Ensure the requested model is present before starting video processing."""
    print(f"Pulling Ollama model: {model}")
    ollama.pull(model=model)
    print(f"Model is ready: {model}")


def unload_model(model: str) -> None:
    """Release the model and its context after one video request."""
    print(f"Unloading Ollama model: {model}")
    try:
        ollama.generate(model=model, keep_alive=0)
    except Exception as exc:
        print(
            f"Could not unload Ollama model '{model}': {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_path = output_path_for_model(args.model)

    if not math.isfinite(args.fps) or args.fps <= 0 or (
        args.context_window is not None and args.context_window < 1
    ):
        print("--fps and --context-window must both be greater than zero", file=sys.stderr)
        return 2
    if not input_dir.is_dir():
        print(f"Input directory does not exist: {input_dir}", file=sys.stderr)
        return 2

    try:
        pull_model(args.model)
    except Exception as exc:
        print(
            f"Could not pull Ollama model '{args.model}': {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1

    video_attempted = False
    try:
        videos = find_videos(input_dir)
        if not videos:
            print(f"No .mp4 files found under: {input_dir}", file=sys.stderr)
            return 2

        rows: list[dict[str, str]] = []
        unavailable_model_message: str | None = None
        for index, video_path in enumerate(videos, start=1):
            relative_path = video_path.relative_to(input_dir).as_posix()
            print(f"[{index}/{len(videos)}] Processing {relative_path}")
            if unavailable_model_message:
                workflow = f"ERROR: Not attempted. {unavailable_model_message}"
                print(f"  {workflow}", file=sys.stderr)
                rows.append(
                    {
                        "video_filename": relative_path,
                        "prompt_execution_seconds": "",
                        "workflow": workflow,
                    }
                )
                continue

            try:
                video_attempted = True
                workflow, elapsed_seconds = analyze_video(
                    video_path, args.model, args.fps, args.context_window
                )
                prompt_execution_seconds = f"{elapsed_seconds:.3f}"
                if not workflow:
                    workflow = "ERROR: Ollama returned an empty response"
                    print(f"  {workflow}", file=sys.stderr)
            except PromptExecutionError as exc:
                original_error = exc.cause
                workflow = f"ERROR: {type(original_error).__name__}: {original_error}"
                prompt_execution_seconds = f"{exc.elapsed_seconds:.3f}"
                print(f"  {workflow}", file=sys.stderr)
                if is_unsupported_mllama_error(original_error):
                    unavailable_model_message = (
                        "The installed Ollama runtime does not support Llama 3.2 Vision's "
                        "'mllama' architecture. Install an Ollama version that supports this model "
                        "before rerunning."
                    )
            except Exception as exc:  # Keep the batch running and preserve the failed row.
                workflow = f"ERROR: {type(exc).__name__}: {exc}"
                prompt_execution_seconds = ""
                print(f"  {workflow}", file=sys.stderr)
                if is_unsupported_mllama_error(exc):
                    unavailable_model_message = (
                        "The installed Ollama runtime does not support Llama 3.2 Vision's "
                        "'mllama' architecture. Install an Ollama version that supports this model "
                        "before rerunning."
                    )
            rows.append(
                {
                    "video_filename": relative_path,
                    "prompt_execution_seconds": prompt_execution_seconds,
                    "workflow": workflow,
                }
            )

        write_csv(rows, output_path)
        print(f"Wrote {len(rows)} rows to {output_path}")
        return 0
    finally:
        if not video_attempted:
            unload_model(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
