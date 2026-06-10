# 🛡️ Bringer: Your Local AI Document Assistant

Bringer is a lightweight, fully local AI assistant that lives on your computer. It reads your private documents (`.pdf`, `.docx`, `.txt`, `.md`) and answers your questions using local Large Language Models (LLMs). 

Unlike other AI tools, **nothing ever leaves your machine**, and Bringer is smart enough to automatically adjust its performance based on your hardware—running faster when plugged into power and saving energy when you're on battery.

---

## 🚀 1. Requirements (The Basics)

Before you begin, make sure your computer has the following:

1.  **Python 3.10 or higher**: [Download here](https://www.python.org/downloads/).
2.  **C++ Build Tools**: Required to run the AI engine on your specific hardware.
    *   **Windows**: Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the **"Desktop development with C++"** workload.
3.  **An NVIDIA GPU (Optional but Recommended)**: If you have an NVIDIA graphics card, Bringer will use it to generate answers much faster.

---

## 📥 2. Getting Started

### Step 1: Install Bringer
Open your terminal (search for "PowerShell" on Windows) and run these commands one by one:

```bash
# 1. Download the project (if using git)
git clone <your-repository-url>
cd Bringer

# 2. Create a "virtual environment" (a private space for Bringer's files)
python -m venv .venv

# 3. Activate the environment
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# 4. Install the application
pip install -e .
```

### Step 2: Initialize Configuration
Bringer needs a configuration file to know which models to use. Create it by running:
```bash
bringer init-config
```

---

## 🧠 3. Setting Up Your AI Models

Bringer uses **GGUF models**. These are single files that contain the "brain" of the AI. You can find many of these on [HuggingFace](https://huggingface.co/models?library=gguf).

### The Three-Profile System
Bringer changes its "brain" based on your power state:
*   **High Performance**: Used when your laptop is plugged in.
*   **Balanced**: Used when you are on battery.
*   **Low Power**: Used when your laptop's "Power Saver" mode is on.

### Assigning Models
Once you have downloaded some `.gguf` files, tell Bringer where they are. Here are the commands for each profile:

```bash
# 1. High Performance (Typically used when your laptop is plugged in)
bringer models set high_performance llm "C:/path/to/your/large-model.gguf"

# 2. Balanced (Typically used when you are running on battery)
bringer models set balanced llm "C:/path/to/your/medium-model.gguf"

# 3. Low Power (Typically used when Power Saver mode is active)
bringer models set low_power llm "C:/path/to/your/small-model.gguf"
```

### Manually Switching Profiles
If you want to manually force Bringer to use a specific profile regardless of your battery state:

```bash
# Force High Performance
bringer models profile high_performance

# Force Balanced
bringer models profile balanced

# Force Low Power
bringer models profile low_power

# Return to Automatic Detection (Recommended)
bringer power auto
```

---

## 📂 4. Using Bringer

### Step 1: Add Your Documents
Copy your PDFs, Word docs, or text files into the `documents` folder inside the Bringer project directory.

### Step 2: Index Your Files
Bringer needs to "read" and index your files so it can search them later. Run this whenever you add new files:
```bash
bringer --reindex
```

### Step 3: Start Chatting!
Now you can start the assistant:
```bash
bringer
```
**Just type your question and press Enter.** Bringer will search your documents, find the relevant sections, and write an answer for you.

---

## 🛠️ 5. Command Reference

Here is a list of everything you can do with the `bringer` command, along with examples:

### Core Commands
| Command | Purpose | Example Usage |
| :--- | :--- | :--- |
| `bringer` | Start the interactive assistant. | `bringer` |
| `bringer --reindex` | Rebuild the search index from your `documents` folder. | `bringer --reindex` |
| `bringer doctor` | Checks if everything is installed and configured correctly. | `bringer doctor` |
| `bringer --status` | Shows how many documents you have indexed. | `bringer --status` |
| `bringer --debug` | Start with detailed logs (useful for troubleshooting). | `bringer --debug` |

### Model Management (`bringer models ...`)
| Command | Purpose | Example Usage |
| :--- | :--- | :--- |
| `bringer models show` | See current model-to-profile mappings. | `bringer models show` |
| `bringer models set <profile> llm <path>` | Assign a model file to a profile. | `bringer models set balanced llm "C:/models/qwen.gguf"` |
| `bringer models reload` | Switch to a new model without restarting. | `bringer models reload` |
| `bringer models scan` | Scans project for `.gguf` files. | `bringer models scan` |
| `bringer models profile <name>` | Manually switch to a specific profile. | `bringer models profile low_power` |

### Power Management (`bringer power ...`)
| Command | Purpose | Example Usage |
| :--- | :--- | :--- |
| `bringer power status` | See battery/GPU state and active profile. | `bringer power status` |
| `bringer power auto` | Enable automatic profile switching. | `bringer power auto` |
| `bringer power low` | Force Bringer into Low Power mode. | `bringer power low` |
| `bringer power balanced` | Force Bringer into Balanced mode. | `bringer power balanced` |
| `bringer power high` | Force Bringer into High Performance mode. | `bringer power high` |

### System Commands
| Command | Purpose | Example Usage |
| :--- | :--- | :--- |
| `bringer install` | Pure validation check of your installation. | `bringer install` |
| `bringer update` | Safely checks for updates while preserving config. | `bringer update` |
| `bringer uninstall` | Removes application but keeps models/config. | `bringer uninstall` |
| `bringer uninstall --purge` | Deletes everything including models and DB. | `bringer uninstall --purge` |
| `bringer init-config` | Creates the initial `models.json` file. | `bringer init-config` |

---

## ❓ 6. Troubleshooting

*   **"llama-cpp not installed"**: Make sure you installed the C++ Build Tools mentioned in Section 1, then run `pip install llama-cpp-python`.
*   **"No valid LLM is loaded"**: You haven't told Bringer where your model files are. Use the `bringer models set` command from Section 3.
*   **Slow Answers**: If you have an NVIDIA GPU, make sure you installed the GPU version of the AI engine. Run:
    `$env:CMAKE_ARGS="-DGGML_CUDA=on"; pip install --upgrade --force-reinstall llama-cpp-python --no-cache-dir`

---

## 🔒 Privacy Notice
Bringer is **100% private**. It does not have an "upload" button because your data never leaves your hard drive. All AI math is done by your computer's own processor.
