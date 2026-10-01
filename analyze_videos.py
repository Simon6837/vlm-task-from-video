"""Describe the workflows in every MP4 under a directory with Ollama Vision.

Llama 3.2 Vision accepts images rather than video files. This script therefore
extracts evenly spaced video frames and attaches those images to one chat
request for each video.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

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
DEFAULT_FRAME_COUNT = 12
DEFAULT_CONTEXT_WINDOW = 16384
PROMPT = "The person in the video is performing a task, Give me their exact workflow"


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
        "--frame-count",
        type=int,
        default=DEFAULT_FRAME_COUNT,
        help=f"Evenly spaced frames to attach per video (default: {DEFAULT_FRAME_COUNT})",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=DEFAULT_CONTEXT_WINDOW,
        help=(
            "Ollama context window in tokens for each request "
            f"(default: {DEFAULT_CONTEXT_WINDOW})"
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


def evenly_spaced_indices(total_frames: int, requested_count: int) -> list[int]:
    """Return up to requested_count unique frame indexes spanning the video."""
    if total_frames <= 0:
        raise ValueError("video reports no frames")

    sample_count = min(total_frames, requested_count)
    if sample_count == 1:
        return [0]
    return [round(index * (total_frames - 1) / (sample_count - 1)) for index in range(sample_count)]


def extract_frames(video_path: Path, destination: Path, frame_count: int) -> list[Path]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError("could not open video")

    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_paths: list[Path] = []
        for output_index, frame_index in enumerate(evenly_spaced_indices(total_frames, frame_count)):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f"could not decode frame {frame_index}")

            frame_path = destination / f"frame_{output_index + 1:02d}.jpg"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"could not write temporary frame {frame_index}")
            frame_paths.append(frame_path)
        return frame_paths
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
    video_path: Path, model: str, frame_count: int, context_window: int
) -> tuple[str, float]:
    with tempfile.TemporaryDirectory(prefix="ollama_video_frames_") as temporary_directory:
        frame_paths = extract_frames(video_path, Path(temporary_directory), frame_count)
        started_at = time.perf_counter()
        try:
            response = ollama.chat(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": PROMPT,
                        "images": [str(frame_path) for frame_path in frame_paths],
                    }
                ],
                options={"num_ctx": context_window},
            )
        except Exception as exc:
            raise PromptExecutionError(exc, time.perf_counter() - started_at) from exc
        elapsed_seconds = time.perf_counter() - started_at
    return response_text(response).strip(), elapsed_seconds


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
    """Release the model from Ollama after the complete batch has finished."""
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

    if args.frame_count < 1 or args.context_window < 1:
        print("--frame-count and --context-window must both be at least 1", file=sys.stderr)
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
                workflow, elapsed_seconds = analyze_video(
                    video_path, args.model, args.frame_count, args.context_window
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
        unload_model(args.model)


if __name__ == "__main__":
    raise SystemExit(main())
