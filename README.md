# Video workflow extraction

This script pulls the Ollama model supplied on the command line, then asks it to describe the workflow shown in every MP4 file below `videos/`.

Ollama vision models accept images, not video files. For each video, the script extracts 12 evenly spaced JPEG frames, attaches them to one Ollama request with the configured prompt, and writes the response to `video_workflows.csv`. The request uses a 16,384-token context window because the 12 images exceed Ollama's 4,096-token default.

## Setup

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run

```powershell
python analyze_videos.py gemma4:e4b
```

The script pulls `gemma4:e4b` automatically, keeps it loaded while processing the batch, then unloads it when processing finishes. It writes `results/gemma4-e4b.csv`. Windows does not allow `:` in filenames, so model-name characters that Windows forbids are replaced with `-`; the model tag is retained. The output CSV has `video_filename`, `prompt_execution_seconds`, and `workflow` columns. `prompt_execution_seconds` measures each Ollama request only, excluding model download and frame extraction. A failed video still receives a row; its workflow value starts with `ERROR:`.

Use `python analyze_videos.py --help` to change the input directory, frame count, or context window.

If the available memory cannot support a 16,384-token context window, use fewer frames instead:

```powershell
python analyze_videos.py gemma4:e4b --frame-count 8 --context-window 8192
```

## Llama 3.2 Vision compatibility

`gemma4:e4b` avoids the Llama 3.2 Vision compatibility issue. This section applies only when selecting `llama3.2-vision` as the positional model: if Ollama reports `unknown model architecture: 'mllama'`, the installed Ollama runtime cannot run that model. Ollama's upstream guidance for this regression is to use Ollama `0.24.0`; see [issue 16490](https://github.com/ollama/ollama/issues/16490).
