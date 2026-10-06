# Video workflow extraction

This script pulls the Ollama model supplied on the command line, then asks it to describe the workflow shown in every MP4 file below `videos/`.

Ollama vision models accept images, not video files. For each video, the script samples frames at 1 FPS by default and combines every sampled frame into one timestamped, five-column contact-sheet JPEG. It attaches that single image to one Ollama request and writes the response to `video_workflows.csv`. By default, the script reserves context for one image and retries once with Ollama's reported prompt-token count if necessary. It unloads the model and clears its context after every video. Use `--context-window` to set a fixed context size explicitly.

Each `workflow` value is a JSON array. Every element represents one distinct action, in chronological order, with `action`, `object`, `target`, and `motion` fields.

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python analyze_videos.py gemma4:e4b
```

The script pulls `gemma4:e4b` automatically and unloads it after each video. It writes `results/gemma4-e4b.csv`. Windows does not allow `:` in filenames, so model-name characters that Windows forbids are replaced with `-`; the model tag is retained. The output CSV has `video_filename`, `prompt_execution_seconds`, and `workflow` columns. `prompt_execution_seconds` measures Ollama request attempts for that video only, excluding model download and frame extraction. A failed video still receives a row; its workflow value starts with `ERROR:`.

Use `python analyze_videos.py --help` to change the input directory, sampling FPS, or context window. Higher FPS adds more tiles to the one contact sheet; for long videos, this makes each tile smaller.

For fast actions, sample more frequently. If needed, set an explicit context window:

```powershell
python analyze_videos.py gemma4:e4b --fps 2 --context-window 8192
```

## Llama 3.2 Vision compatibility

`gemma4:e4b` avoids the Llama 3.2 Vision compatibility issue. This section applies only when selecting `llama3.2-vision` as the positional model: if Ollama reports `unknown model architecture: 'mllama'`, the installed Ollama runtime cannot run that model. Ollama's upstream guidance for this regression is to use Ollama `0.24.0`; see [issue 16490](https://github.com/ollama/ollama/issues/16490).
