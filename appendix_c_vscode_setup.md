[← Back to Table of Contents](./README.md)

# Appendix C — VSCode Advanced Setup

> "The right tool, properly configured, makes you an order of magnitude more productive." — Every senior engineer who has watched a junior manually format code.

This appendix is the definitive VSCode setup guide for ML engineers. Follow it end-to-end and you will have a professional-grade development environment in under an hour.

---

## C.1 Essential Extensions

Install these extensions for a complete ML development experience:

| Extension | Publisher | Purpose |
|---|---|---|
| **Python** | Microsoft | Python language support, IntelliSense |
| **Pylance** | Microsoft | Fast, rich Python type checking |
| **Ruff** | Astral Software | Extremely fast linter + formatter |
| **Black Formatter** | Microsoft | Opinionated Python formatter |
| **GitLens** | GitKraken | Git blame, history, and annotations |
| **GitHub Copilot** | GitHub | AI pair programmer |
| **GitHub Copilot Chat** | GitHub | AI chat assistant in editor |
| **Jupyter** | Microsoft | Notebook support in VSCode |
| **Docker** | Microsoft | Dockerfile/Compose support |
| **Remote - SSH** | Microsoft | Develop on remote GPU servers |
| **Dev Containers** | Microsoft | Develop inside Docker containers |
| **Todo Tree** | Gruntfuggly | Find and list TODOs across project |
| **Error Lens** | Alexander | Inline error/warning highlights |
| **indent-rainbow** | oderwat | Colorize indentation levels |
| **YAML** | Red Hat | YAML language support |

### Install via Command Line

```bash
# Install all recommended extensions at once
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension charliermarsh.ruff
code --install-extension ms-python.black-formatter
code --install-extension eamodio.gitlens
code --install-extension GitHub.copilot
code --install-extension GitHub.copilot-chat
code --install-extension ms-toolsai.jupyter
code --install-extension ms-azuretools.vscode-docker
code --install-extension ms-vscode-remote.remote-ssh
code --install-extension ms-vscode-remote.remote-containers
code --install-extension Gruntfuggly.todo-tree
code --install-extension usernamehw.errorlens
code --install-extension oderwat.indent-rainbow
code --install-extension redhat.vscode-yaml
```

### Team-Shared Extension Recommendations

```json
// .vscode/extensions.json — commit this to your repo
{
    "recommendations": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff",
        "ms-python.black-formatter",
        "eamodio.gitlens",
        "GitHub.copilot",
        "GitHub.copilot-chat",
        "ms-toolsai.jupyter",
        "ms-azuretools.vscode-docker",
        "Gruntfuggly.todo-tree",
        "usernamehw.errorlens"
    ],
    "unwantedRecommendations": []
}
```

---

## C.2 settings.json — Complete Recommended Configuration

### User Settings (Global)

```json
// Open with: Ctrl+Shift+P → "Preferences: Open User Settings (JSON)"
{
    // ─── Editor ────────────────────────────────────
    "editor.fontSize": 14,
    "editor.fontFamily": "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
    "editor.fontLigatures": true,
    "editor.lineHeight": 1.6,
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "editor.rulers": [88, 120],
    "editor.wordWrap": "off",
    "editor.minimap.enabled": false,
    "editor.bracketPairColorization.enabled": true,
    "editor.guides.bracketPairs": "active",
    "editor.stickyScroll.enabled": true,
    "editor.inlineSuggest.enabled": true,
    "editor.cursorBlinking": "smooth",
    "editor.cursorSmoothCaretAnimation": "on",
    "editor.renderWhitespace": "trailing",
    "editor.suggestSelection": "first",

    // ─── Formatting ────────────────────────────────
    "editor.formatOnSave": true,
    "editor.formatOnPaste": false,
    "editor.codeActionsOnSave": {
        "source.organizeImports": "explicit",
        "source.fixAll": "explicit"
    },

    // ─── Python ────────────────────────────────────
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.tabSize": 4,
        "editor.rulers": [88, 120],
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },

    // ─── Ruff ──────────────────────────────────────
    "ruff.lint.run": "onSave",
    "ruff.fixAll": true,
    "ruff.organizeImports": true,

    // ─── Files ─────────────────────────────────────
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "files.trimFinalNewlines": true,
    "files.autoSave": "afterDelay",
    "files.autoSaveDelay": 1000,
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/*.pyc": true,
        "**/.mypy_cache": true,
        "**/.ruff_cache": true,
        "**/.egg-info": true
    },
    "files.associations": {
        "*.yml": "yaml",
        "Dockerfile*": "dockerfile",
        "*.jsonl": "json",
        "requirements*.txt": "pip-requirements"
    },

    // ─── Terminal ──────────────────────────────────
    "terminal.integrated.fontSize": 13,
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.scrollback": 10000,
    "terminal.integrated.copyOnSelection": true,

    // ─── Git ───────────────────────────────────────
    "git.autofetch": true,
    "git.confirmSync": false,
    "git.enableSmartCommit": true,
    "gitlens.codeLens.enabled": true,

    // ─── Copilot ───────────────────────────────────
    "github.copilot.enable": {
        "*": true,
        "yaml": true,
        "markdown": true,
        "plaintext": false
    },

    // ─── Misc ──────────────────────────────────────
    "workbench.colorTheme": "One Dark Pro",
    "workbench.iconTheme": "material-icon-theme",
    "workbench.startupEditor": "none",
    "explorer.confirmDelete": false,
    "explorer.confirmDragAndDrop": false,
    "telemetry.telemetryLevel": "off"
}
```

### Workspace Settings (Per-Project)

```json
// .vscode/settings.json — commit to each project repo
{
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv/bin/python",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff",
        "editor.formatOnSave": true,
        "editor.codeActionsOnSave": {
            "source.fixAll.ruff": "explicit",
            "source.organizeImports.ruff": "explicit"
        }
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "ruff.configuration": "${workspaceFolder}/pyproject.toml",
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/wandb": true,
        "**/outputs": true
    },
    "search.exclude": {
        "**/data": true,
        "**/checkpoints": true,
        "**/*.bin": true,
        "**/*.pt": true
    }
}
```

---

## C.3 Keyboard Shortcuts

### Most Productive Shortcuts

| Action | Windows/Linux | macOS |
|---|---|---|
| **Command Palette** | `Ctrl+Shift+P` | `⌘+Shift+P` |
| **Quick File Open** | `Ctrl+P` | `⌘+P` |
| **Global Search** | `Ctrl+Shift+F` | `⌘+Shift+F` |
| **Go to Symbol** | `Ctrl+Shift+O` | `⌘+Shift+O` |
| **Go to Definition** | `F12` | `F12` |
| **Peek Definition** | `Alt+F12` | `⌥+F12` |
| **Find References** | `Shift+F12` | `Shift+F12` |
| **Rename Symbol** | `F2` | `F2` |
| **Toggle Terminal** | `` Ctrl+` `` | `` ⌘+` `` |
| **Toggle Sidebar** | `Ctrl+B` | `⌘+B` |
| **Split Editor** | `Ctrl+\` | `⌘+\` |
| **Multi-Cursor (click)** | `Alt+Click` | `⌥+Click` |
| **Multi-Cursor (line)** | `Ctrl+Alt+↑/↓` | `⌘+⌥+↑/↓` |
| **Select Word** | `Ctrl+D` | `⌘+D` |
| **Select All Occurrences** | `Ctrl+Shift+L` | `⌘+Shift+L` |
| **Move Line Up/Down** | `Alt+↑/↓` | `⌥+↑/↓` |
| **Duplicate Line** | `Shift+Alt+↑/↓` | `Shift+⌥+↑/↓` |
| **Delete Line** | `Ctrl+Shift+K` | `⌘+Shift+K` |
| **Comment Line** | `Ctrl+/` | `⌘+/` |
| **Fold/Unfold** | `Ctrl+Shift+[/]` | `⌘+⌥+[/]` |
| **Quick Fix** | `Ctrl+.` | `⌘+.` |
| **Format Document** | `Shift+Alt+F` | `Shift+⌥+F` |
| **Zen Mode** | `Ctrl+K Z` | `⌘+K Z` |

### Custom Keybindings

```json
// Open with: Ctrl+Shift+P → "Preferences: Open Keyboard Shortcuts (JSON)"
[
    {
        "key": "ctrl+shift+r",
        "command": "python.execInTerminal",
        "when": "editorTextFocus && editorLangId == python"
    },
    {
        "key": "ctrl+shift+t",
        "command": "python.testing.runCurrentFile",
        "when": "editorTextFocus && editorLangId == python"
    },
    {
        "key": "ctrl+shift+d",
        "command": "python.testing.debugCurrentFile",
        "when": "editorTextFocus && editorLangId == python"
    }
]
```

---

## C.4 Debugging — launch.json Configurations

### Python Script

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Current File",
            "type": "debugpy",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "justMyCode": true,
            "env": {
                "CUDA_VISIBLE_DEVICES": "0"
            }
        },
        {
            "name": "Python: Train",
            "type": "debugpy",
            "request": "launch",
            "program": "${workspaceFolder}/train.py",
            "args": [
                "--config", "configs/debug.yaml",
                "--epochs", "2",
                "--batch-size", "4"
            ],
            "console": "integratedTerminal",
            "justMyCode": false,
            "env": {
                "CUDA_VISIBLE_DEVICES": "0",
                "WANDB_MODE": "disabled"
            }
        },
        {
            "name": "Python: Pytest",
            "type": "debugpy",
            "request": "launch",
            "module": "pytest",
            "args": [
                "tests/",
                "-v",
                "--tb=short",
                "-x"
            ],
            "console": "integratedTerminal",
            "justMyCode": false
        },
        {
            "name": "Python: FastAPI",
            "type": "debugpy",
            "request": "launch",
            "module": "uvicorn",
            "args": [
                "app.main:app",
                "--reload",
                "--host", "0.0.0.0",
                "--port", "8000"
            ],
            "console": "integratedTerminal",
            "justMyCode": true
        },
        {
            "name": "Python: Attach Remote",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 5678
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/app"
                }
            ]
        }
    ]
}
```

### Remote Debugging Setup

```python
# Add to your training script for remote debugging
# pip install debugpy
import debugpy

debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger to attach...")
debugpy.wait_for_client()
print("Debugger attached!")
```

```bash
# On the remote server
python train.py

# On your local machine — forward the debug port
ssh -L 5678:localhost:5678 gpu-server

# Then in VSCode: Run → "Python: Attach Remote"
```

---

## C.5 Remote Development

### Remote-SSH for GPU Servers

<div class="diagram">
<div class="diagram-title">Remote-SSH Development Flow</div>
<div class="flow-h">
<div class="flow-node blue">Local VSCode</div>
<div class="flow-arrow">→ SSH →</div>
<div class="flow-node green">Remote GPU Server</div>
<div class="flow-arrow">→</div>
<div class="flow-node purple">Files, Terminal, Debugger</div>
</div>
</div>

```bash
# 1. Install Remote-SSH extension

# 2. Configure SSH (see Appendix B for ~/.ssh/config)

# 3. Connect: Ctrl+Shift+P → "Remote-SSH: Connect to Host"

# 4. Select your host from ~/.ssh/config

# 5. VSCode opens a new window connected to the remote machine
#    - File explorer shows remote files
#    - Terminal runs on remote machine
#    - Extensions run on remote machine (install them there too)
#    - Debugger works on remote code
```

**Recommended Remote-SSH settings:**
```json
{
    "remote.SSH.remotePlatform": {
        "gpu-server": "linux"
    },
    "remote.SSH.connectTimeout": 30,
    "remote.SSH.defaultExtensions": [
        "ms-python.python",
        "ms-python.vscode-pylance",
        "charliermarsh.ruff"
    ]
}
```

### Dev Containers for Reproducible Environments

```json
// .devcontainer/devcontainer.json
{
    "name": "ML Development",
    "build": {
        "dockerfile": "Dockerfile"
    },
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.vscode-pylance",
                "charliermarsh.ruff",
                "ms-toolsai.jupyter",
                "GitHub.copilot"
            ],
            "settings": {
                "python.defaultInterpreterPath": "/opt/venv/bin/python",
                "editor.formatOnSave": true
            }
        }
    },
    "features": {
        "ghcr.io/devcontainers/features/git:1": {},
        "ghcr.io/devcontainers/features/github-cli:1": {}
    },
    "forwardPorts": [8888, 6006, 8000],
    "postCreateCommand": "pip install -e '.[dev]'",
    "runArgs": ["--gpus", "all"]
}
```

```dockerfile
# .devcontainer/Dockerfile
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip git curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /workspace
```

---

## C.6 GitHub Copilot & AI

### Setup

```json
// settings.json
{
    "github.copilot.enable": {
        "*": true,
        "yaml": true,
        "markdown": true
    },
    "github.copilot.editor.enableAutoCompletions": true
}
```

### Effective Prompting Patterns

```python
# Pattern 1: Descriptive function signature → Copilot fills the body
def calculate_bleu_score(reference: list[str], hypothesis: list[str]) -> float:
    """Calculate BLEU score between reference and hypothesis sentences."""
    # Copilot auto-completes here

# Pattern 2: Comment-driven development
# Load a pretrained BERT model and tokenizer for sentiment classification
# then create a function that takes a list of texts and returns sentiment labels

# Pattern 3: Type hints guide better suggestions
def preprocess_dataset(
    raw_data: pd.DataFrame,
    text_column: str,
    label_column: str,
    max_length: int = 512,
    test_size: float = 0.2,
) -> tuple[Dataset, Dataset]:
    """Split and tokenize dataset for training."""

# Pattern 4: Test-driven — write the test, let Copilot write the code
def test_normalize_text():
    assert normalize_text("Hello  WORLD!") == "hello world!"
    assert normalize_text("  spaces  ") == "spaces"
    assert normalize_text("") == ""
```

### Copilot Chat Commands

| Command | Purpose |
|---|---|
| `/explain` | Explain selected code |
| `/fix` | Fix errors in selected code |
| `/tests` | Generate tests for selected code |
| `/doc` | Generate documentation |
| `@workspace` | Ask about the entire workspace |
| `@terminal` | Ask about terminal output |

---

## C.7 Git Integration

### Source Control View

```
Ctrl+Shift+G — Open Source Control panel

Features:
- See all changed files at a glance
- Stage individual files or hunks
- Write commit messages inline
- Push/pull from the status bar
- Branch management
```

### Merge Conflict Resolution

VSCode provides an inline merge editor:

```
<<<<<<< HEAD (Current Change)
learning_rate = 0.001
=======
learning_rate = 0.0001
>>>>>>> feature-branch (Incoming Change)

Buttons appear above the conflict:
- Accept Current Change
- Accept Incoming Change
- Accept Both Changes
- Compare Changes
```

### GitLens Features

```json
// Recommended GitLens settings
{
    "gitlens.codeLens.enabled": true,
    "gitlens.codeLens.authors.enabled": true,
    "gitlens.currentLine.enabled": true,
    "gitlens.hovers.currentLine.over": "line",
    "gitlens.blame.format": "${author|10} ${date} ${message|50}"
}
```

> **Tip**: Hover over any line to see who last modified it, when, and in which commit. Use `Alt+B` to toggle full file blame.

---

## C.8 Jupyter Integration

### Running Notebooks in VSCode

```
1. Open any .ipynb file — rendered natively
2. Ctrl+Shift+P → "Jupyter: Create New Notebook"
3. Select kernel (your .venv Python)
4. Run cells with Shift+Enter
```

### Interactive Window

```python
# Add "# %%" to create cells in a regular .py file
# Then run them interactively with Shift+Enter

# %% [markdown]
# # My Experiment
# This is a markdown cell

# %%
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
df.head()

# %%
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["loss"])
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# %%
# Variable Explorer shows all variables in the interactive session
# Click the grid icon in the notebook toolbar to open it
```

> **Pro tip**: Use `# %%` cells in `.py` files instead of notebooks for version control. You get the interactivity of notebooks with the version-control-friendliness of plain Python.

---

## C.9 Productivity Tips

### Multi-Cursor Editing

```
Alt+Click           — Add cursor at click position
Ctrl+Alt+↑/↓       — Add cursor above/below
Ctrl+D              — Select next occurrence of selection
Ctrl+Shift+L        — Select ALL occurrences of selection
Ctrl+U              — Undo last cursor operation

Example: Rename a variable in 10 places:
1. Select the variable name
2. Ctrl+Shift+L (selects all occurrences)
3. Type the new name (all update simultaneously)
```

### Snippets

```json
// .vscode/python.code-snippets — project-specific snippets
{
    "ML Training Loop": {
        "prefix": "trainloop",
        "body": [
            "for epoch in range(${1:num_epochs}):",
            "    model.train()",
            "    for batch in ${2:train_loader}:",
            "        optimizer.zero_grad()",
            "        outputs = model(batch)",
            "        loss = ${3:criterion}(outputs, batch['labels'])",
            "        loss.backward()",
            "        optimizer.step()",
            "    ",
            "    print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')"
        ],
        "description": "PyTorch training loop boilerplate"
    },
    "Dataclass Config": {
        "prefix": "mlconfig",
        "body": [
            "from dataclasses import dataclass, field",
            "",
            "",
            "@dataclass",
            "class ${1:TrainingConfig}:",
            "    \"\"\"${2:Configuration for training.}\"\"\"",
            "",
            "    learning_rate: float = ${3:1e-4}",
            "    batch_size: int = ${4:32}",
            "    num_epochs: int = ${5:10}",
            "    seed: int = ${6:42}"
        ],
        "description": "Python dataclass for ML config"
    },
    "Pytest Function": {
        "prefix": "testfn",
        "body": [
            "def test_${1:function_name}():",
            "    \"\"\"Test ${2:description}.\"\"\"",
            "    # Arrange",
            "    ${3:input_val} = ${4:None}",
            "",
            "    # Act",
            "    result = ${5:function}(${3:input_val})",
            "",
            "    # Assert",
            "    assert result == ${6:expected}"
        ],
        "description": "Pytest test function with AAA pattern"
    }
}
```

### Tasks

```json
// .vscode/tasks.json — run common commands from VSCode
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Train Model",
            "type": "shell",
            "command": "python train.py --config configs/base.yaml",
            "group": "build",
            "presentation": {
                "reveal": "always",
                "panel": "new"
            }
        },
        {
            "label": "Run Tests",
            "type": "shell",
            "command": "pytest tests/ -v --tb=short",
            "group": "test",
            "presentation": {
                "reveal": "always"
            }
        },
        {
            "label": "Lint",
            "type": "shell",
            "command": "ruff check . && mypy src/",
            "group": "build"
        },
        {
            "label": "Start TensorBoard",
            "type": "shell",
            "command": "tensorboard --logdir outputs/ --port 6006",
            "isBackground": true,
            "presentation": {
                "reveal": "silent"
            }
        }
    ]
}
```

### Workspace Trust

```json
// When opening untrusted repos, VSCode restricts extensions
// For your own projects, trust the workspace:
// Ctrl+Shift+P → "Workspaces: Manage Workspace Trust"

// Or in settings:
{
    "security.workspace.trust.enabled": true,
    "security.workspace.trust.untrustedFiles": "prompt"
}
```

---

## C.10 Complete .vscode Directory Template

For any new ML project, create this structure:

```
.vscode/
├── settings.json          # Project-specific settings
├── extensions.json        # Recommended extensions
├── launch.json            # Debug configurations
├── tasks.json             # Build/test/lint tasks
└── python.code-snippets   # Project snippets
```

<div class="diagram">
<div class="diagram-title">VSCode ML Development Flow</div>
<div class="cycle">
<div class="cycle-step blue">Write Code (Copilot assists)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step green">Auto-format on Save (Ruff)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step purple">Debug (launch.json)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step orange">Test (pytest integration)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step teal">Commit (GitLens + Source Control)</div>
<div class="cycle-arrow">→</div>
<div class="cycle-step accent">Deploy (Remote-SSH / Containers)</div>
<div class="cycle-arrow">↩</div>
</div>
</div>

> **The golden rule**: Automate everything that can be automated. Format-on-save, lint-on-save, organize-imports-on-save. Your focus should be on the ML problem, not on code style.

---

*Last updated: April 2026*
