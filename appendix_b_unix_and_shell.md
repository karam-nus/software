[← Back to Table of Contents](./README.md)

# Appendix B — Unix & Shell Essentials

> "The Unix philosophy: Write programs that do one thing and do it well. Write programs to work together." — Doug McIlroy

This appendix is your daily reference for the Unix/Linux command line — the environment where most ML training, deployment, and data processing happens. Master these tools and you will be dramatically more productive.

---

## B.1 Shell Basics

### Navigating the Filesystem

```bash
# Print current directory
pwd

# List files (short)
ls

# List files (detailed, hidden, human-readable sizes)
ls -lah

# List sorted by modification time (newest first)
ls -lt

# Change directory
cd /home/user/projects
cd ~            # home directory
cd -            # previous directory
cd ..           # parent directory

# Create directories (including parents)
mkdir -p projects/ml-experiment/data

# Find where a command lives
which python
type python
```

### File Operations

```bash
# Copy file
cp model.pt model_backup.pt

# Copy directory recursively
cp -r experiment_01/ experiment_02/

# Move / rename
mv old_name.py new_name.py
mv results/ archive/results_2024/

# Remove file
rm unwanted_file.txt

# Remove directory recursively (CAREFUL!)
rm -rf old_experiment/

# Create an empty file / update timestamp
touch new_file.py

# Create a symlink (common for datasets)
ln -s /data/shared/imagenet ./data/imagenet
```

### Viewing File Contents

```bash
# Print entire file
cat config.yaml

# First / last N lines
head -20 training.log
tail -50 training.log

# Follow a log file in real-time (essential for monitoring training)
tail -f training.log

# Page through a file
less large_file.csv        # q to quit, / to search, n for next match

# Count lines, words, characters
wc -l dataset.csv          # line count
wc -w document.txt         # word count

# Search within files
grep "accuracy" training.log
grep -r "import torch" src/          # recursive search
grep -n "error" training.log         # show line numbers
grep -i "warning" training.log       # case-insensitive
grep -c "epoch" training.log         # count matches
grep -v "DEBUG" training.log         # invert match (exclude)
```

---

## B.2 File Permissions

```bash
# View permissions
ls -la
# drwxr-xr-x  user group  4096  Mar 1  src/
# -rw-r--r--  user group  2048  Mar 1  train.py

# Permission bits: rwx = read(4) write(2) execute(1)
# Three groups: owner | group | others

# Make a script executable
chmod +x train.sh

# Set specific permissions (owner: rwx, group: rx, others: rx)
chmod 755 train.sh

# Set file to read-only
chmod 444 config.yaml

# Common permission patterns
chmod 644 file.txt     # owner rw, others r  (typical file)
chmod 755 script.sh    # owner rwx, others rx (typical script)
chmod 600 id_rsa       # owner rw only        (SSH keys)
chmod 700 .ssh/        # owner rwx only       (SSH directory)

# Change owner
sudo chown user:group file.txt
sudo chown -R user:group directory/

# Default permission mask
umask 022    # new files: 644, new dirs: 755
umask 077    # new files: 600, new dirs: 700 (private)
```

---

## B.3 Process Management

```bash
# List your processes
ps aux | grep python

# Interactive process viewer
top                    # basic
htop                   # better (install: sudo apt install htop)

# GPU processes (essential for ML)
nvidia-smi
watch -n 1 nvidia-smi   # refresh every second

# Run in background
python train.py &

# Redirect output and run in background
python train.py > train.log 2>&1 &

# nohup — survives terminal disconnection
nohup python train.py > train.log 2>&1 &

# Job control
jobs                   # list background jobs
fg %1                  # bring job 1 to foreground
bg %1                  # resume job 1 in background
# Ctrl+Z               # suspend foreground process

# Kill processes
kill 12345             # send SIGTERM (graceful)
kill -9 12345          # send SIGKILL (force — last resort)

# Find process ID
ps aux | grep "train.py"
pgrep -f "train.py"

# Kill all GPU processes (when GPU is stuck)
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill
```

---

## B.4 SSH — Remote Access

### Key Generation and Setup

```bash
# Generate an SSH key pair (Ed25519 — modern and secure)
ssh-keygen -t ed25519 -C "your.email@example.com"

# Start ssh-agent and add key
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

# Copy public key to remote server
ssh-copy-id user@gpu-server.example.com

# Test connection
ssh -T git@github.com
```

### SSH Config File

```bash
# ~/.ssh/config — essential for managing multiple servers
Host gpu-server
    HostName 10.0.1.50
    User mluser
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host gpu-cluster-*
    User mluser
    IdentityFile ~/.ssh/id_ed25519
    ProxyJump bastion

Host bastion
    HostName bastion.example.com
    User mluser
    IdentityFile ~/.ssh/id_ed25519

# Now you can just type:
# ssh gpu-server
# instead of:
# ssh -i ~/.ssh/id_ed25519 mluser@10.0.1.50
```

### Port Forwarding

```bash
# Local port forwarding — access remote Jupyter on local browser
ssh -L 8888:localhost:8888 gpu-server
# Then open http://localhost:8888 in your browser

# Local port forwarding — access remote TensorBoard
ssh -L 6006:localhost:6006 gpu-server

# Remote port forwarding — expose local service to remote
ssh -R 9090:localhost:9090 gpu-server

# Dynamic SOCKS proxy
ssh -D 1080 gpu-server

# Multiple port forwards
ssh -L 8888:localhost:8888 -L 6006:localhost:6006 gpu-server
```

### File Transfer

```bash
# SCP — simple copy
scp model.pt gpu-server:/home/mluser/models/
scp gpu-server:/home/mluser/results.csv ./
scp -r experiment/ gpu-server:/home/mluser/experiments/

# rsync — better for large transfers (incremental, resumable)
rsync -avz --progress data/ gpu-server:/data/project/
rsync -avz --progress gpu-server:/results/ ./results/

# rsync with exclusions
rsync -avz --exclude '.git' --exclude '__pycache__' \
    --exclude '*.pyc' --exclude '.venv' \
    ./project/ gpu-server:/home/mluser/project/

# Resume interrupted transfer
rsync -avz --partial --progress large_model.pt gpu-server:/models/
```

---

## B.5 Text Processing

### grep — Search

```bash
# Search for a pattern
grep "learning_rate" config.yaml

# Recursive search with context
grep -rn -A2 -B2 "RuntimeError" src/

# Search with regex
grep -E "epoch [0-9]+" training.log

# Find files containing a pattern
grep -rl "import torch" src/

# Search only Python files
grep -rn --include="*.py" "def forward" src/
```

### sed — Stream Editor

```bash
# Replace text (in-place)
sed -i 's/learning_rate: 0.001/learning_rate: 0.0001/' config.yaml

# Replace all occurrences on each line
sed -i 's/old_text/new_text/g' file.txt

# Delete lines matching a pattern
sed -i '/^#/d' config.txt          # remove comments
sed -i '/^$/d' config.txt          # remove blank lines

# Print specific lines
sed -n '10,20p' large_file.csv     # lines 10-20

# Insert text before a line
sed -i '5i\new_line_of_text' file.txt
```

### awk — Column Processing

```bash
# Print specific columns
awk '{print $1, $3}' results.tsv

# Filter rows by condition
awk '$3 > 0.95 {print $0}' results.tsv

# Sum a column
awk '{sum += $2} END {print "Total:", sum}' data.tsv

# Custom field separator
awk -F',' '{print $1, $NF}' data.csv

# Process training logs — extract epoch and loss
grep "loss:" training.log | awk '{print $2, $4}'
```

### sort, uniq, cut, tr, xargs

```bash
# Sort numerically by second column
sort -t',' -k2 -n results.csv

# Count unique values
cut -d',' -f3 data.csv | sort | uniq -c | sort -rn

# Replace characters
echo "hello world" | tr ' ' '\n'        # space to newline
cat file.txt | tr '[:upper:]' '[:lower:]'  # lowercase

# Convert DOS line endings to Unix
tr -d '\r' < windows_file.txt > unix_file.txt

# xargs — build commands from stdin
find . -name "*.pyc" | xargs rm
cat gpu_servers.txt | xargs -I{} ssh {} "nvidia-smi"

# Parallel execution with xargs
find data/ -name "*.json" | xargs -P4 -I{} python process.py {}
```

---

## B.6 Shell Scripting

### Variables and Quoting

```bash
#!/usr/bin/env bash

# Variables
PROJECT_NAME="ml-experiment"
EPOCHS=100
LEARNING_RATE=0.001
DATA_DIR="/data/${PROJECT_NAME}"

# Command substitution
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
GIT_HASH=$(git rev-parse --short HEAD)
GPU_COUNT=$(nvidia-smi -L | wc -l)

# String interpolation
echo "Running ${PROJECT_NAME} at ${TIMESTAMP} on ${GPU_COUNT} GPUs"
```

### Conditionals

```bash
#!/usr/bin/env bash
set -euo pipefail    # exit on error, undefined vars, pipe failures

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU available"
    DEVICE="cuda"
else
    echo "No GPU — using CPU"
    DEVICE="cpu"
fi

# Check if file exists
if [[ -f "checkpoint.pt" ]]; then
    echo "Resuming from checkpoint"
    RESUME_FLAG="--resume checkpoint.pt"
else
    RESUME_FLAG=""
fi

# Check if directory exists
[[ -d "data/" ]] || mkdir -p "data/"

# Numeric comparison
if (( EPOCHS > 50 )); then
    echo "Long training run"
fi
```

### Loops

```bash
#!/usr/bin/env bash

# Loop over hyperparameters
for LR in 0.001 0.0001 0.00001; do
    for BATCH_SIZE in 16 32 64; do
        echo "Training with lr=${LR}, bs=${BATCH_SIZE}"
        python train.py --lr "$LR" --batch-size "$BATCH_SIZE" \
            --output "results/lr${LR}_bs${BATCH_SIZE}/"
    done
done

# Loop over files
for FILE in data/*.csv; do
    echo "Processing ${FILE}"
    python process.py --input "$FILE"
done

# While loop — wait for GPU
while [[ $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -gt 1000 ]]; do
    echo "GPU busy, waiting..."
    sleep 60
done
echo "GPU free — starting training"
```

### Functions

```bash
#!/usr/bin/env bash

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log "ERROR: nvidia-smi not found"
        return 1
    fi
    local gpu_count
    gpu_count=$(nvidia-smi -L | wc -l)
    log "Found ${gpu_count} GPU(s)"
    return 0
}

run_training() {
    local config_file=$1
    local output_dir=$2
    mkdir -p "$output_dir"
    log "Starting training: config=${config_file} output=${output_dir}"
    python train.py --config "$config_file" --output "$output_dir" 2>&1 | tee "${output_dir}/train.log"
}

# Usage
check_gpu || exit 1
run_training "configs/base.yaml" "outputs/run_$(date +%s)"
```

### Pipes and Redirection

```bash
# Pipe output to next command
cat training.log | grep "epoch" | tail -10

# Redirect stdout to file
python train.py > output.log

# Redirect stdout and stderr
python train.py > output.log 2>&1

# Redirect stderr only
python train.py 2> errors.log

# Append to file
echo "new result" >> results.txt

# Here document
cat > config.yaml << 'EOF'
model:
  name: resnet50
  pretrained: true
training:
  epochs: 100
  lr: 0.001
EOF

# Process substitution — diff two command outputs
diff <(sort file1.txt) <(sort file2.txt)
```

---

## B.7 tmux — Terminal Multiplexer

tmux lets you run persistent terminal sessions that survive disconnections — essential for long-running ML training on remote servers.

### Essential Commands

```bash
# Start a new named session
tmux new -s training

# Detach from session: Ctrl+b, then d

# List sessions
tmux ls

# Reattach to session
tmux attach -t training

# Kill a session
tmux kill-session -t training
```

### tmux Cheat Sheet

| Action | Shortcut |
|---|---|
| **Sessions** | |
| Detach | `Ctrl+b` `d` |
| List sessions | `Ctrl+b` `s` |
| Rename session | `Ctrl+b` `$` |
| **Windows** (tabs) | |
| New window | `Ctrl+b` `c` |
| Next window | `Ctrl+b` `n` |
| Previous window | `Ctrl+b` `p` |
| Select window by number | `Ctrl+b` `0-9` |
| Rename window | `Ctrl+b` `,` |
| Close window | `Ctrl+b` `&` |
| **Panes** (splits) | |
| Horizontal split | `Ctrl+b` `"` |
| Vertical split | `Ctrl+b` `%` |
| Switch pane | `Ctrl+b` `arrow` |
| Close pane | `Ctrl+b` `x` |
| Resize pane | `Ctrl+b` `Ctrl+arrow` |
| Toggle zoom (full-screen pane) | `Ctrl+b` `z` |
| **Scroll / Copy** | |
| Enter scroll mode | `Ctrl+b` `[` |
| Exit scroll mode | `q` |
| Search backward | `Ctrl+b` `[` then `?` |

### Recommended tmux Config

```bash
# ~/.tmux.conf
set -g mouse on                    # enable mouse
set -g history-limit 50000         # scrollback buffer
set -g default-terminal "screen-256color"
set -g status-interval 5

# Better prefix
unbind C-b
set -g prefix C-a
bind C-a send-prefix

# Easy pane splitting
bind | split-window -h -c "#{pane_current_path}"
bind - split-window -v -c "#{pane_current_path}"

# Status bar
set -g status-right '#{?window_zoomed_flag,🔍,} %H:%M %d-%b'
```

### Training Session Workflow

```bash
# Start a training session
tmux new -s train

# Split into panes: training | GPU monitor | logs
# Pane 1: run training
python train.py --config config.yaml

# Ctrl+b % (vertical split)
# Pane 2: monitor GPU
watch -n 1 nvidia-smi

# Ctrl+b " (horizontal split)
# Pane 3: follow logs
tail -f outputs/training.log

# Detach with Ctrl+b d — training continues!
# Reattach later: tmux attach -t train
```

---

## B.8 Environment Configuration

### PATH and Environment Variables

```bash
# View PATH
echo $PATH

# Add to PATH (current session)
export PATH="$HOME/.local/bin:$PATH"

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="my-experiment"
export HF_HOME="$HOME/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# Check a variable
echo $CUDA_VISIBLE_DEVICES
printenv | grep CUDA
```

### ~/.bashrc / ~/.zshrc

```bash
# ~/.bashrc or ~/.zshrc — runs on every new shell

# Aliases
alias ll='ls -lah'
alias gs='git status'
alias gd='git diff'
alias gl='git log --oneline -20'
alias py='python'
alias jl='jupyter lab'
alias ns='nvidia-smi'
alias nsw='watch -n 1 nvidia-smi'

# ML-specific aliases
alias ta='tmux attach -t'
alias tl='tmux ls'
alias tn='tmux new -s'

# Conda initialization
eval "$(/opt/conda/bin/conda shell.bash hook)"

# Default CUDA device
export CUDA_VISIBLE_DEVICES=0

# Larger tmux scrollback
export HISTSIZE=100000
export HISTFILESIZE=200000
```

---

## B.9 Cron Jobs — Scheduling Tasks

```bash
# Edit crontab
crontab -e

# Format: minute hour day month weekday command
# ┌──── minute (0-59)
# │ ┌──── hour (0-23)
# │ │ ┌──── day of month (1-31)
# │ │ │ ┌──── month (1-12)
# │ │ │ │ ┌──── day of week (0-7, 0=Sun)
# │ │ │ │ │
# * * * * * command

# Run training every night at 2 AM
0 2 * * * cd /home/user/project && /home/user/.venv/bin/python train.py >> /home/user/logs/cron.log 2>&1

# Clean old checkpoints every Sunday at 3 AM
0 3 * * 0 find /home/user/checkpoints -mtime +30 -name "*.pt" -delete

# Run evaluation every 6 hours
0 */6 * * * cd /home/user/project && bash eval.sh >> /home/user/logs/eval.log 2>&1

# GPU health check every 5 minutes
*/5 * * * * nvidia-smi > /dev/null 2>&1 || echo "GPU error at $(date)" >> /home/user/logs/gpu.log

# List current crontab
crontab -l
```

---

## B.10 Disk & Network

```bash
# Disk usage
df -h                          # filesystem overview
du -sh *                       # size of each item in current dir
du -sh data/                   # size of a directory
du -sh --max-depth=1 /home/    # top-level breakdown

# Find large files
find . -type f -size +1G -exec ls -lh {} \;

# Network — check open ports
ss -tlnp                       # listening TCP ports

# Download files
wget https://example.com/dataset.tar.gz
curl -O https://example.com/dataset.tar.gz
curl -L -o model.bin "https://huggingface.co/model/resolve/main/model.bin"

# Download with resume
wget -c https://example.com/large_file.tar.gz

# Open files by process
lsof -i :8080                  # what's using port 8080

# Compress and extract
tar -czf archive.tar.gz directory/     # compress
tar -xzf archive.tar.gz               # extract
zip -r archive.zip directory/          # zip
unzip archive.zip                      # unzip
```

---

## B.11 ML Job Submission Script

A complete shell script combining many of the above concepts:

```bash
#!/usr/bin/env bash
set -euo pipefail

# ─── Configuration ─────────────────────────────
EXPERIMENT_NAME=${1:-"default"}
CONFIG_FILE=${2:-"configs/base.yaml"}
NUM_GPUS=${3:-1}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/${EXPERIMENT_NAME}_${TIMESTAMP}"

# ─── Preflight checks ─────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Starting experiment: ${EXPERIMENT_NAME}"

if ! command -v nvidia-smi &> /dev/null; then
    log "ERROR: nvidia-smi not found"; exit 1
fi

GPU_COUNT=$(nvidia-smi -L | wc -l)
if (( NUM_GPUS > GPU_COUNT )); then
    log "ERROR: Requested ${NUM_GPUS} GPUs but only ${GPU_COUNT} available"
    exit 1
fi

if [[ ! -f "$CONFIG_FILE" ]]; then
    log "ERROR: Config file not found: ${CONFIG_FILE}"; exit 1
fi

# ─── Setup ─────────────────────────────────────
mkdir -p "$OUTPUT_DIR"
cp "$CONFIG_FILE" "${OUTPUT_DIR}/config.yaml"
git rev-parse HEAD > "${OUTPUT_DIR}/git_hash.txt"

log "Output directory: ${OUTPUT_DIR}"
log "Using ${NUM_GPUS} GPU(s)"

# ─── Launch training ──────────────────────────
export CUDA_VISIBLE_DEVICES=$(seq -s',' 0 $((NUM_GPUS - 1)))

if (( NUM_GPUS > 1 )); then
    torchrun --nproc_per_node="$NUM_GPUS" \
        train.py --config "$CONFIG_FILE" --output "$OUTPUT_DIR" \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
else
    python train.py --config "$CONFIG_FILE" --output "$OUTPUT_DIR" \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

log "Training complete! Results in ${OUTPUT_DIR}"
```

```bash
# Usage
chmod +x run_experiment.sh
./run_experiment.sh my_experiment configs/large.yaml 4
```

---

*Last updated: April 2026*
