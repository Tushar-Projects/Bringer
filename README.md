# Bringer

Bringer is a local, CLI-based AI assistant for your documents. Point it at a folder, let it index the files, and ask questions against that knowledge base without sending anything to a cloud API.

It is built for people who want a practical offline RAG workflow: simple to run, grounded in source documents, and transparent about where answers came from.

## Features

- CLI-first workflow that stays fast and simple in the terminal
- Automatic document indexing from the local `documents/` folder
- Hybrid retrieval using semantic search plus keyword search
- Source-grounded answers with file names and page numbers where available
- Fully local operation with LM Studio and no cloud dependency
- Dynamic model selection based on available hardware
- Clean default output with an optional `--debug` mode for deeper visibility

## Installation

### Quick start

```powershell
git clone <repo>
cd bringer
pip install -e .
Bringer
```

### Windows shortcut

If you are on Windows, you can also run:

```powershell
install.bat
```

That script upgrades `pip`, installs Bringer in editable mode, and leaves you ready to launch it with `Bringer`.

## Prerequisites

Before running Bringer, make sure you have:

- Python 3.10 or newer
- [LM Studio](https://lmstudio.ai/)
- The `lms` CLI available in your terminal
- At least one supported local model installed in LM Studio

Bringer currently selects from these locally installed models:

- `Qwen2.5-7B-Instruct-1M-Q6_K`
- `qwen2.5-3b-instruct`
- `gemma-4-E2B-it-Q4_K_M`

You can confirm what is installed with:

```powershell
lms ls
```

## Usage

Bringer supports a small set of CLI commands:

```powershell
Bringer
Bringer --help
Bringer --status
Bringer --reindex
Bringer --debug
```

### What each command does

- `Bringer` starts the assistant
- `Bringer --help` shows the help menu
- `Bringer --status` shows indexed files, chunk counts, and LM Studio model status
- `Bringer --reindex` clears and rebuilds the document index
- `Bringer --debug` starts Bringer with detailed internal logs

## Example

```text
> What is a watchdog timer?

Answer:
"A watchdog timer is an electronic or software timer used to detect system failures."

Sources:
- watchdog_test.txt (page 1)
```

The goal is not to sound clever. The goal is to give you the closest grounded answer Bringer can find in your files, and show where it came from.

## How It Works

At a high level, Bringer follows a pretty simple pipeline:

1. Documents are read from the `documents/` folder of the project and broken into smaller chunks.
2. Those chunks are indexed for both semantic search and keyword search.
3. When you ask a question, Bringer retrieves the most relevant matches from both methods.
4. The results are reranked so the strongest evidence rises to the top.
5. The local LLM answers using that context and returns the response with source citations.

Under the hood, that gives you a balance between flexible retrieval and grounded answers, without needing any hosted service.

## Design Choices

### Local-first

Bringer is meant to run on your machine, with your files, using your local model setup. That keeps the workflow private and avoids the cost or friction of external APIs.

### CLI instead of a GUI

The project started as a command-line tool on purpose. A terminal interface is quick to launch, easy to automate, and honest about what the system is doing.

### Hybrid retrieval

Pure vector search can miss obvious keyword matches. Pure keyword search can miss semantically related text. Bringer combines both so retrieval is more resilient in real document collections.

## Hardware-Aware Model Selection

Bringer keeps model selection automatic:

- GPU available and plugged in -> `Qwen2.5-7B-Instruct-1M-Q6_K`
- GPU available but on battery -> `qwen2.5-3b-instruct`
- CPU only -> `gemma-4-E2B-it-Q4_K_M`

That helps it stay usable across different machines without making the user pick a model every time.

## Limitations

Bringer works well when the documents are clean and the relevant information is actually present, but it is still important to be realistic about its limits:

- It depends heavily on document quality and extractable text
- Smaller local models can miss nuance or answer less precisely
- LM Studio still needs to be installed and configured correctly
- Complex questions can still surface imperfect retrieval or partial answers

In other words, Bringer is best treated as a grounded document assistant, not as an authority beyond the source material.

## Future Improvements

- A lightweight GUI for people who prefer not to work in the terminal
- Better line-level extraction for even tighter source grounding
- Faster startup and indexing on larger document collections

## Project Structure

- [`bringer_cli.py`](/D:/proj%20Bringer/Bringer/bringer_cli.py): main CLI entry point
- [`config.py`](/D:/proj%20Bringer/Bringer/config.py): core configuration
- [`src/modules/`](/D:/proj%20Bringer/Bringer/src/modules): retrieval, indexing, LM Studio, and hardware logic
- [`documents/`](/D:/proj%20Bringer/Bringer/documents): files Bringer indexes
- [`vector_db/`](/D:/proj%20Bringer/Bringer/vector_db): local vector database storage
- [`tests/`](/D:/proj%20Bringer/Bringer/tests): unit tests

## Notes

- Bringer is designed to work fully offline once your local dependencies and models are installed.
- The default CLI output is intentionally clean.
- If you want to inspect retrieval behavior, timings, or loading details, run `Bringer --debug`.
