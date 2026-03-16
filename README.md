# Bringer

Bringer is a fully local Retrieval-Augmented Generation (RAG) assistant that indexes your documents, retrieves relevant context, and answers questions using a local LM Studio model.

It is designed for a single-machine workflow:

- local document ingestion
- local embeddings and reranking
- local vector database
- local LM Studio inference
- clean CLI experience by default, with `--debug` for full internal logs

## What Bringer Does

Bringer watches a local `documents/` folder, extracts text from supported files, chunks and embeds the content, stores vectors in ChromaDB, and answers questions against that indexed knowledge base.

At runtime it:

1. Detects your hardware
2. Chooses an installed local model
3. Ensures LM Studio is running and the correct model is loaded
4. Retrieves relevant chunks with hybrid semantic + keyword search
5. Reranks results
6. Sends grounded context to the local LLM
7. Streams an answer with source citations

## Features

- Fully local RAG pipeline
- LM Studio integration with automatic model loading
- Hardware-aware model selection
- Persistent local vector database via ChromaDB
- Hybrid retrieval:
  semantic vector search + BM25 keyword search
- Cross-encoder reranking
- Automatic document reindexing through a file watcher
- Clean CLI output by default
- `--debug` mode for detailed internal logs

## Supported Document Types

Bringer currently supports:

- `.pdf`
- `.docx`
- `.pptx`
- `.txt`
- `.md`

## Prerequisites

Before installing Bringer, make sure you have the following available on your machine.

### 1. Python

- Python `3.10` is recommended

The repository currently includes a `.python-version`, and the test environment in this project has been using Python `3.10.x`.

### 2. LM Studio

You need:

- [LM Studio](https://lmstudio.ai/)
- the `lms` CLI available in your system `PATH`
- the LM Studio local server feature enabled

Bringer talks to LM Studio through:

- `http://localhost:1234/v1`

### 3. Installed Local Models

Bringer is currently configured to use only these locally installed models:

- `qwen2.5-coder-7b-instruct`
- `qwen2.5-3b-instruct`
- `qwen2.5-coder-1.5b-instruct`

You can confirm your installed models with:

```powershell
lms ls
```

### 4. Optional GPU Support

Bringer can run on CPU, but it is designed to benefit from NVIDIA GPUs when available.

GPU detection uses:

- PyTorch CUDA detection
- `nvidia-smi` as a fallback

## Hardware-Based Model Selection

Bringer selects the model automatically based on the detected hardware state:

- GPU available and plugged in -> `qwen2.5-coder-7b-instruct`
- GPU available and on battery -> `qwen2.5-3b-instruct`
- CPU only -> `qwen2.5-coder-1.5b-instruct`

This logic is implemented in [`src/modules/hardware_detector.py`](/D:/proj%20Bringer/Bringer/src/modules/hardware_detector.py).

## Project Structure

Key files and folders:

- [`bringer_cli.py`](/D:/proj%20Bringer/Bringer/bringer_cli.py)
  main CLI entry point
- [`config.py`](/D:/proj%20Bringer/Bringer/config.py)
  global configuration
- [`src/modules/`](/D:/proj%20Bringer/Bringer/src/modules)
  core RAG modules
- [`documents/`](/D:/proj%20Bringer/Bringer/documents)
  source documents to index
- [`vector_db/`](/D:/proj%20Bringer/Bringer/vector_db)
  persistent ChromaDB storage
- [`logs/`](/D:/proj%20Bringer/Bringer/logs)
  local logs
- [`tests/`](/D:/proj%20Bringer/Bringer/tests)
  unit tests

## Installation

### 1. Open the project folder

```powershell
cd "D:\proj Bringer\Bringer"
```

### 2. Create a virtual environment

```powershell
python -m venv .venv
```

### 3. Activate the virtual environment

PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```powershell
pip install -r requirements.txt
pip install -e .
```

The editable install exposes the console command:

```powershell
Bringer
```

## Configuration

Main configuration lives in [`config.py`](/D:/proj%20Bringer/Bringer/config.py).

Notable settings:

- `LLM_API_BASE`
- `LLM_MODEL_LARGE`
- `LLM_MODEL_MEDIUM`
- `LLM_MODEL_SMALL`
- `EMBEDDING_MODEL_NAME`
- `RERANKER_MODEL_NAME`
- `FORCE_CPU`
- `DEBUG_MODE`

Important:

- Bringer is currently set up to talk to LM Studio at `http://localhost:1234/v1`
- `DEBUG_MODE` defaults to `False`
- You can also enable debug output with `Bringer --debug`

## LM Studio Setup

Before running Bringer for the first time:

1. Install LM Studio
2. Install the supported local models
3. Ensure the `lms` CLI works from your terminal
4. Make sure the LM Studio local server can run on port `1234`

Bringer will:

- start LM Studio if it is not already running
- load the selected model with the `Bringer_RAG` preset
- stop LM Studio on shutdown only if Bringer started the server

Useful commands:

```powershell
lms ls
lms server start
lms server stop
```

## How to Use Bringer

### 1. Add your documents

Put supported files into:

- [`documents/`](/D:/proj%20Bringer/Bringer/documents)

Bringer watches this folder and updates the vector database automatically.

### 2. Start the assistant

```powershell
Bringer
```

Example clean startup:

```text
Bringer AI Assistant

Model: qwen2.5-coder-7b-instruct (GPU)
Documents indexed: 71

Ready.
Ask a question or type 'exit'.
```

### 3. Ask questions

Example:

```text
> What is a watchdog timer?
```

Example response shape:

```text
A watchdog timer is a hardware or software timer used to detect system failures.

Sources
- Module_2.pdf (page 17 | chunk 3)
- watchdog_test.txt (chunk 0)
```

### 4. Exit

Type either:

- `exit`
- `quit`

Or press `Ctrl+C`.

Shutdown output:

```text
Shutting down Bringer...
```

## Debug Mode

To enable detailed logs:

```powershell
Bringer --debug
```

Debug mode keeps the internal pipeline output visible, including:

- hardware detection details
- LM Studio startup and model loading logs
- retrieval diagnostics
- prompt/token diagnostics
- model and indexing status logs

This is useful for development, troubleshooting, and performance tuning.

## How Indexing Works

When a file is added or modified in `documents/`, Bringer:

1. Loads the file contents
2. Normalizes and chunks the text
3. Generates embeddings
4. Stores the vectors in ChromaDB
5. Rebuilds retrieval state as needed

Bringer also tracks file hashes so unchanged files are not unnecessarily reprocessed.

## How Answering Works

For each query, Bringer:

1. Optionally expands the query
2. Runs hybrid retrieval
3. Merges semantic and keyword candidates
4. Reranks the results
5. Builds a grounded prompt
6. Streams a response from LM Studio
7. Appends source citations

## Running Tests

Activate the virtual environment first:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\.venv\Scripts\Activate.ps1
```

Run the current unit tests:

```powershell
python -m unittest tests.test_bringer_cli tests.test_lmstudio_manager tests.test_hardware_detector -v
```

## Troubleshooting

### `Bringer` command not found

Make sure you installed the project in editable mode:

```powershell
pip install -e .
```

### LM Studio does not start

Check:

- LM Studio is installed
- `lms` is available in `PATH`
- port `1234` is not blocked

Try:

```powershell
lms server start
```

### Model fails to load

Check that the model exists locally:

```powershell
lms ls
```

Only these models are currently expected:

- `qwen2.5-coder-7b-instruct`
- `qwen2.5-3b-instruct`
- `qwen2.5-coder-1.5b-instruct`

### GPU is not detected

Bringer checks both:

- PyTorch CUDA
- `nvidia-smi`

If GPU detection still fails, verify:

- NVIDIA drivers are installed
- `nvidia-smi` works in the terminal
- your Python environment is valid

### PowerShell blocks virtual environment activation

Use a process-scoped bypass:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
. .\.venv\Scripts\Activate.ps1
```

### No results are returned

Check:

- your files are inside `documents/`
- the files were parsed successfully
- the vector database has indexed content
- your query matches the document content well enough

## Development Notes

- The CLI entry point is registered in [`setup.py`](/D:/proj%20Bringer/Bringer/setup.py)
- The local vector database is persisted in [`vector_db/`](/D:/proj%20Bringer/Bringer/vector_db)
- The system is designed to run fully locally
- Default CLI output is intentionally clean; use `--debug` for internals

## Quick Start

```powershell
cd "D:\proj Bringer\Bringer"
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
Bringer
```

## License

No license file is currently included in this repository.
