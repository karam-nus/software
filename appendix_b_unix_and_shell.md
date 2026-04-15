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

## B.5 Passwordless SSH from Windows to Remote Linux

Setting up passwordless (key-based) SSH access from a Windows machine to a remote Linux server is one of the most common tasks for ML engineers — whether the server is on your office intranet or an AWS EC2 instance on the internet. This section covers both scenarios end-to-end.

<div class="diagram">
<div class="diagram-title">SSH Key Authentication Flow</div>
<div class="flow-h">
  <div class="flow-node accent">🖥️ Windows PC <small>Private key stays here</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green">🔐 SSH Handshake <small>Public key challenge</small></div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue">🐧 Linux Server <small>Public key in authorized_keys</small></div>
</div>
</div>

### Prerequisites — Enable OpenSSH on Windows

Windows 10/11 ships with OpenSSH built in. Verify it works:

```bash
# Open PowerShell (or Windows Terminal) and check
ssh -V
# → OpenSSH_for_Windows_9.x ...

# If not found, install via Settings → Apps → Optional Features → OpenSSH Client
# Or via PowerShell (admin):
Add-WindowsOptionalFeature -Online -FeatureName OpenSSH.Client
```

> **Tip**: Always use **Windows Terminal** or **PowerShell** — not the legacy Command Prompt. All commands below work in both PowerShell and Git Bash.

### Scenario A — Intranet Linux Server (Password-Based Initially)

This is the most common case: your GPU server is at `10.0.1.50` on the office network, and you currently log in with a password.

**Step 1 — Generate an SSH key pair on Windows**

```bash
# Open PowerShell on your Windows PC
ssh-keygen -t ed25519 -C "yourname@company.com"

# When prompted:
#   Enter file: press Enter for default (C:\Users\YourName\.ssh\id_ed25519)
#   Enter passphrase: press Enter for no passphrase (or set one for extra security)

# This creates two files:
#   C:\Users\YourName\.ssh\id_ed25519       ← PRIVATE key (never share)
#   C:\Users\YourName\.ssh\id_ed25519.pub   ← PUBLIC key (copy to server)
```

**Step 2 — Copy the public key to the Linux server**

```bash
# Method 1: Using ssh-copy-id (if available in Git Bash)
ssh-copy-id mluser@10.0.1.50

# Method 2: Manual copy (works in PowerShell — no ssh-copy-id needed)
# First, display your public key:
type $env:USERPROFILE\.ssh\id_ed25519.pub

# Then SSH in with password one last time and add the key:
ssh mluser@10.0.1.50

# On the Linux server, run:
mkdir -p ~/.ssh && chmod 700 ~/.ssh
echo "PASTE_YOUR_PUBLIC_KEY_HERE" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
exit
```

```bash
# Method 3: One-liner from PowerShell (copies key in a single command)
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh mluser@10.0.1.50 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

**Step 3 — Test passwordless login**

```bash
# From PowerShell — should connect without asking for a password
ssh mluser@10.0.1.50
```

**Step 4 — Set up SSH config for convenience**

```bash
# Create/edit: C:\Users\YourName\.ssh\config
# (In PowerShell: notepad $env:USERPROFILE\.ssh\config)

Host gpu-server
    HostName 10.0.1.50
    User mluser
    IdentityFile C:\Users\YourName\.ssh\id_ed25519
    ForwardAgent yes
    ServerAliveInterval 60
    ServerAliveCountMax 3

Host gpu-server-2
    HostName 10.0.1.51
    User mluser
    IdentityFile C:\Users\YourName\.ssh\id_ed25519
```

```bash
# Now you can simply type:
ssh gpu-server

# Port forwarding for Jupyter also becomes cleaner:
ssh -L 8888:localhost:8888 gpu-server
```

### Scenario B — AWS EC2 Instance (with .pem Key File)

When you launch an EC2 instance, AWS gives you a `.pem` private key file. This replaces key generation — AWS already placed the matching public key on the instance.

<div class="diagram">
<div class="diagram-title">AWS EC2 SSH Flow</div>
<div class="flow">
  <div class="flow-node orange wide">📥 Download .pem from AWS Console</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node accent wide">🔒 Set correct permissions on .pem</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node green wide">🔑 SSH with -i flag pointing to .pem</div>
  <div class="flow-arrow accent"></div>
  <div class="flow-node blue wide">☁️ Connected to EC2 instance</div>
</div>
</div>

**Step 1 — Move .pem file to your .ssh directory**

```bash
# Move the downloaded .pem file
# In PowerShell:
Move-Item "$env:USERPROFILE\Downloads\my-gpu-instance.pem" "$env:USERPROFILE\.ssh\my-gpu-instance.pem"
```

**Step 2 — Fix permissions (critical on Windows)**

On Linux/macOS you'd run `chmod 400`. On Windows, you must restrict the file's ACL so only your user can read it — otherwise SSH refuses to use the key.

```powershell
# PowerShell — remove inherited permissions and grant only your user
$keyPath = "$env:USERPROFILE\.ssh\my-gpu-instance.pem"

# Remove all existing access rules
icacls $keyPath /inheritance:r

# Grant only your user read access
icacls $keyPath /grant "${env:USERNAME}:(R)"

# Verify — should show only your username
icacls $keyPath
```

```bash
# If using Git Bash or WSL instead, the standard Unix command works:
chmod 400 ~/.ssh/my-gpu-instance.pem
```

**Step 3 — Connect to EC2**

```bash
# Direct connection (replace with your instance's public IP/DNS)
ssh -i C:\Users\YourName\.ssh\my-gpu-instance.pem ubuntu@ec2-54-123-45-67.compute-1.amazonaws.com

# Or with public IP directly
ssh -i C:\Users\YourName\.ssh\my-gpu-instance.pem ubuntu@54.123.45.67
```

> **Note**: The default username depends on the AMI — `ubuntu` for Ubuntu, `ec2-user` for Amazon Linux, `admin` for Debian.

**Step 4 — Add to SSH config (so you never type that again)**

```bash
# Add to C:\Users\YourName\.ssh\config

Host aws-gpu
    HostName 54.123.45.67
    User ubuntu
    IdentityFile C:\Users\YourName\.ssh\my-gpu-instance.pem
    ForwardAgent yes
    ServerAliveInterval 60
    StrictHostKeyChecking no

# If you have a bastion/jump host in your VPC:
Host aws-gpu-private
    HostName 10.0.1.100
    User ubuntu
    IdentityFile C:\Users\YourName\.ssh\my-gpu-instance.pem
    ProxyJump aws-bastion

Host aws-bastion
    HostName 54.123.45.68
    User ubuntu
    IdentityFile C:\Users\YourName\.ssh\my-gpu-instance.pem
```

```bash
# Now simply:
ssh aws-gpu

# Port forward Jupyter from EC2:
ssh -L 8888:localhost:8888 aws-gpu

# Copy model to EC2:
scp model.pt aws-gpu:/home/ubuntu/models/

# Rsync project to EC2:
rsync -avz --exclude '.git' ./project/ aws-gpu:/home/ubuntu/project/
```

### Scenario C — Using Your Own Key with AWS (instead of .pem)

If you prefer using the same `id_ed25519` key everywhere (rather than per-instance `.pem` files):

```bash
# 1. Generate your key (if you haven't already — see Scenario A, Step 1)
ssh-keygen -t ed25519 -C "yourname@company.com"

# 2. When launching an EC2 instance, choose "Import key pair" in the
#    AWS Console → EC2 → Key Pairs, and upload your id_ed25519.pub

# 3. Or, if the instance already exists, add your key manually:
#    First, connect with the .pem file one time:
ssh -i ~/.ssh/my-gpu-instance.pem ubuntu@54.123.45.67

#    On the EC2 instance:
echo "YOUR_ED25519_PUBLIC_KEY" >> ~/.ssh/authorized_keys

#    Now you can connect with your own key:
ssh ubuntu@54.123.45.67
```

### Troubleshooting

| Problem | Solution |
|---|---|
| `Permission denied (publickey)` | Check key permissions, verify correct username, ensure public key is in `authorized_keys` |
| `.pem` file "too open" error | Fix permissions with `icacls` (Windows) or `chmod 400` (Git Bash/WSL) |
| `Connection refused` | Verify SSH is running on server (`sudo systemctl status sshd`), check security group (AWS) allows port 22 |
| `Connection timed out` | Server is unreachable — check IP, VPN, AWS security group inbound rules |
| `Host key verification failed` | Remove old entry: `ssh-keygen -R hostname`, or set `StrictHostKeyChecking no` in config |
| Key works in Git Bash but not PowerShell | Ensure `ssh-agent` service is running: `Get-Service ssh-agent | Set-Service -StartupType Automatic; Start-Service ssh-agent` |

### Starting ssh-agent on Windows (Persistent)

```powershell
# Run in PowerShell as Administrator (one-time setup)
Get-Service ssh-agent | Set-Service -StartupType Automatic
Start-Service ssh-agent

# Add your key to the agent (so you don't need to specify -i every time)
ssh-add $env:USERPROFILE\.ssh\id_ed25519

# Or add the .pem file
ssh-add $env:USERPROFILE\.ssh\my-gpu-instance.pem

# List added keys
ssh-add -l
```

---

## B.6 Passwordless Git Operations (SSH-Based)

Every time you `git push` or `git pull` and get asked for a password, you're wasting time. Set up SSH keys for Git once and never type credentials again.

<div class="diagram">
<div class="diagram-title">Git Authentication Methods</div>
<div class="compare">
  <div class="compare-side left">
    <div class="compare-title">❌ HTTPS (Password/Token)</div>
    <ul>
      <li>git clone https://github.com/...</li>
      <li>Prompts for username/token on every push</li>
      <li>Must manage personal access tokens</li>
      <li>Token can expire, needs renewal</li>
    </ul>
  </div>
  <div class="compare-side right">
    <div class="compare-title">✅ SSH (Key-Based)</div>
    <ul>
      <li>git clone git@github.com:...</li>
      <li>Never prompted — key handles auth</li>
      <li>One-time setup, works forever</li>
      <li>Same key works for all repos</li>
    </ul>
  </div>
</div>
</div>

### Setup for GitHub

**Step 1 — Generate an SSH key (if you don't have one)**

```bash
# On Linux/macOS:
ssh-keygen -t ed25519 -C "your.email@example.com"

# On Windows (PowerShell):
ssh-keygen -t ed25519 -C "your.email@example.com"
# Default path: C:\Users\YourName\.ssh\id_ed25519
```

**Step 2 — Add the public key to GitHub**

```bash
# Display your public key
# Linux/macOS:
cat ~/.ssh/id_ed25519.pub

# Windows (PowerShell):
type $env:USERPROFILE\.ssh\id_ed25519.pub

# Copy the entire output, then:
# 1. Go to github.com → Settings → SSH and GPG keys → New SSH key
# 2. Title: "My Work Laptop" (or any descriptive name)
# 3. Key type: Authentication Key
# 4. Paste the public key
# 5. Click "Add SSH key"
```

**Step 3 — Test the connection**

```bash
ssh -T git@github.com
# → Hi username! You've been authenticated, but GitHub does not provide shell access.
# This means it's working!
```

**Step 4 — Clone repos using SSH URL (not HTTPS)**

```bash
# SSH URL format (use this):
git clone git@github.com:your-org/ml-project.git

# HTTPS URL format (avoid this):
# git clone https://github.com/your-org/ml-project.git
```

### Switch Existing Repos from HTTPS to SSH

If you already cloned a repo via HTTPS and get prompted for credentials:

```bash
# Check current remote URL
git remote -v
# → origin  https://github.com/your-org/ml-project.git (fetch)
# → origin  https://github.com/your-org/ml-project.git (push)

# Switch to SSH
git remote set-url origin git@github.com:your-org/ml-project.git

# Verify
git remote -v
# → origin  git@github.com:your-org/ml-project.git (fetch)
# → origin  git@github.com:your-org/ml-project.git (push)

# Test — should push without prompting for credentials
git push
```

### Setup for GitLab (Self-Hosted or gitlab.com)

```bash
# Same key generation — your id_ed25519 key works for multiple services

# Add key to GitLab:
# Go to GitLab → Preferences → SSH Keys → Add new key
# Paste the contents of id_ed25519.pub

# Test connection
ssh -T git@gitlab.com
# For self-hosted: ssh -T git@gitlab.yourcompany.com

# Clone with SSH
git clone git@gitlab.com:your-org/ml-project.git
# Self-hosted: git clone git@gitlab.yourcompany.com:your-org/ml-project.git
```

### Multiple Git Accounts (Personal + Work)

If you have different SSH keys for personal and work GitHub/GitLab accounts:

```bash
# ~/.ssh/config (Linux/macOS) or C:\Users\YourName\.ssh\config (Windows)

# Personal GitHub account
Host github.com-personal
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_personal

# Work GitHub account
Host github.com-work
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work

# Work GitLab (self-hosted)
Host gitlab.work
    HostName gitlab.yourcompany.com
    User git
    IdentityFile ~/.ssh/id_ed25519_work
```

```bash
# Clone using the alias from your SSH config:
git clone git@github.com-personal:myuser/side-project.git
git clone git@github.com-work:company/ml-platform.git
git clone git@gitlab.work:team/training-pipeline.git

# The SSH config tells Git which key to use for each host alias
```

### Setting Up on a Remote Linux Server

When you SSH into a GPU server, you also want passwordless Git there (to push results, pull code, etc.). You have two options:

**Option A — SSH Agent Forwarding (recommended)**

Your local key is "forwarded" to the remote server — no need to copy keys.

```bash
# In your SSH config (local machine), ensure ForwardAgent is on:
Host gpu-server
    HostName 10.0.1.50
    User mluser
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes     # ← This forwards your local SSH key

# Now when you SSH in:
ssh gpu-server

# On the remote server, your GitHub key is available:
ssh -T git@github.com
# → Hi username! You've been authenticated ...

# Git operations work without any key setup on the server:
git clone git@github.com:your-org/ml-project.git
git push origin main
```

> **Security note**: Only enable `ForwardAgent` for servers you trust. A compromised server could use your forwarded key.

**Option B — Deploy Key (per-repo, read-only by default)**

For automated systems or shared servers where agent forwarding isn't appropriate:

```bash
# On the remote server, generate a dedicated key:
ssh-keygen -t ed25519 -C "gpu-server-deploy" -f ~/.ssh/deploy_key

# Add the public key as a Deploy Key:
# GitHub → Repo → Settings → Deploy keys → Add deploy key
# Paste the contents of deploy_key.pub
# Check "Allow write access" if the server needs to push

# Configure Git to use this key for the specific repo:
# In ~/.ssh/config on the server:
Host github-deploy
    HostName github.com
    User git
    IdentityFile ~/.ssh/deploy_key

# Clone using the alias:
git clone git@github-deploy:your-org/ml-project.git
```

### Git Config for Clean Commits

After setting up SSH, configure your Git identity so commits are properly attributed:

```bash
# Set globally (applies to all repos)
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# Or per-repo (for work vs personal)
cd ~/work/ml-project
git config user.name "Your Name"
git config user.email "your.work@company.com"

# Verify
git config --list --show-origin | grep user
```

### Troubleshooting

| Problem | Solution |
|---|---|
| `git@github.com: Permission denied (publickey)` | Key not added to GitHub, or ssh-agent not running |
| `ssh -T git@github.com` hangs | Firewall blocking port 22 — try SSH over HTTPS port: `ssh -T -p 443 git@ssh.github.com` |
| Agent forwarding not working | Check `ForwardAgent yes` in config, and `AllowAgentForwarding yes` on server's `/etc/ssh/sshd_config` |
| Wrong account used for push | Check SSH config aliases, use `ssh -T git@github.com-work` to verify which account connects |
| `Could not open connection to auth agent` | Start the agent: `eval "$(ssh-agent -s)"` then `ssh-add` |
| Key works for clone but not push | Deploy key might be read-only — enable write access in GitHub repo settings |

### SSH over HTTPS Port (Firewall Bypass)

Some corporate networks block port 22. GitHub supports SSH over port 443:

```bash
# Test if port 443 works
ssh -T -p 443 git@ssh.github.com

# If it works, add to SSH config:
Host github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/id_ed25519

# Now all git operations use port 443 transparently
git clone git@github.com:your-org/ml-project.git   # uses port 443
```

---

## B.7 Text Processing

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

## B.8 Shell Scripting

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

## B.9 tmux — Terminal Multiplexer

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

## B.10 Environment Configuration

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

## B.11 Cron Jobs — Scheduling Tasks

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

## B.12 Disk & Network

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

## B.13 ML Job Submission Script

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

## B.14 rsync — Fast File Synchronisation

`rsync` is the standard tool for transferring and syncing large files between local and remote machines. It sends only changed bytes, making repeated transfers of large datasets or checkpoints very efficient.

```bash
# Basic syntax
rsync [options] source destination

# Copy a local directory to a remote server (preserve permissions, compress, progress)
rsync -avzP data/ user@server:/data/project/

# Sync a remote checkpoint directory back to local
rsync -avzP user@server:/outputs/run_01/ ./outputs/run_01/

# Mirror a directory (delete files on destination that no longer exist on source)
rsync -avz --delete data/ user@server:/data/project/

# Dry run — show what would be transferred without doing it
rsync -avzn data/ user@server:/data/project/

# Use a non-standard SSH port
rsync -avzP -e "ssh -p 2222" data/ user@server:/data/

# Exclude patterns
rsync -avz --exclude="*.pyc" --exclude="__pycache__/" src/ user@server:/app/src/

# Include only specific file types
rsync -avz --include="*.pt" --exclude="*" checkpoints/ user@server:/checkpoints/

# Throttle bandwidth (KB/s) — useful on shared connections
rsync -avzP --bwlimit=50000 datasets/ user@server:/datasets/
```

### Common rsync Flags

| Flag | Meaning |
|---|---|
| `-a` | Archive mode (preserves permissions, timestamps, symlinks) |
| `-v` | Verbose output |
| `-z` | Compress data during transfer |
| `-P` | Show progress + keep partial files on interruption |
| `-n` | Dry run (no changes made) |
| `--delete` | Remove destination files not in source |
| `--exclude` | Skip matching files/directories |
| `--bwlimit=N` | Limit bandwidth to N KB/s |

### ML Workflow Examples

```bash
# Push latest checkpoint to remote storage
rsync -avzP outputs/run_$(date +%Y%m%d)/ user@storage:/checkpoints/

# Pull a shared dataset once — skip if already synced
rsync -avz --ignore-existing /data/shared/ ./data/

# Sync an entire experiment directory after training completes
rsync -avzP user@gpu-server:/experiments/my_run/ ./experiments/my_run/
```

---

## B.15 jq — JSON Processing

`jq` is the standard command-line tool for parsing, filtering, and transforming JSON — essential for working with API responses, Hugging Face model cards, and config files.

```bash
# Install
sudo apt install jq          # Ubuntu/Debian
brew install jq              # macOS

# Pretty-print JSON
cat response.json | jq '.'
curl https://api.example.com/data | jq '.'

# Extract a field
echo '{"loss": 0.42, "epoch": 10}' | jq '.loss'
# → 0.42

# Extract nested field
jq '.training.learning_rate' config.json

# Extract from an array
jq '.[0]' results.json          # first element
jq '.[-1]' results.json         # last element
jq '.[] | .accuracy' results.json   # all accuracy values

# Filter array by condition
jq '.[] | select(.accuracy > 0.95)' results.json

# Extract multiple fields as new object
jq '{name: .model_name, acc: .accuracy}' results.json

# Get keys of an object
jq 'keys' config.json

# Array length
jq 'length' results.json

# Compact output (no whitespace) — useful for piping
jq -c '.' response.json

# Update a value (non-destructive — prints modified JSON)
jq '.training.epochs = 200' config.json > config_new.json

# Iterate and format as plain text (-r strips quotes)
jq -r '.[] | "\(.name): \(.score)"' results.json
```

### Practical ML Examples

```bash
# Parse Hugging Face model info
curl -s "https://huggingface.co/api/models/bert-base-uncased" | jq '{id: .id, downloads: .downloads}'

# Extract all loss values from a JSON-lines log
cat training_log.jsonl | jq -r '.loss' | paste -sd',' -

# Count experiments with accuracy above threshold
cat results.json | jq '[.[] | select(.val_acc > 0.90)] | length'

# Build a summary table
cat results.json | jq -r '.[] | [.run_id, .val_acc, .epochs] | @tsv'
```

---

## B.16 fzf — Fuzzy Finder

`fzf` is an interactive fuzzy search tool for files, command history, and arbitrary lists. It dramatically speeds up navigation in large codebases and long command histories.

```bash
# Install
sudo apt install fzf          # Ubuntu/Debian
brew install fzf              # macOS
git clone --depth 1 https://github.com/junegunn/fzf.git ~/.fzf && ~/.fzf/install  # manual

# Enable shell integration (add to ~/.bashrc or ~/.zshrc)
[ -f ~/.fzf.bash ] && source ~/.fzf.bash   # bash
[ -f ~/.fzf.zsh ]  && source ~/.fzf.zsh    # zsh
```

### Key Bindings (after shell integration)

| Shortcut | Action |
|---|---|
| `Ctrl+r` | Fuzzy search command history |
| `Ctrl+t` | Fuzzy search files and paste selection into command line |
| `Alt+c` | Fuzzy search directories and `cd` into selection |

### Command-Line Usage

```bash
# Interactively pick a file to open
vim $(fzf)

# Fuzzy search only Python files
fzf --include="*.py"

# Preview file contents while searching
fzf --preview 'cat {}'

# Preview with syntax highlighting (requires bat)
fzf --preview 'bat --color=always {}'

# Search from a list of strings
echo -e "train\neval\ntest" | fzf

# Kill a process interactively
kill -9 $(ps aux | fzf | awk '{print $2}')

# Checkout a git branch interactively
git checkout $(git branch | fzf)

# Open a recently modified file
vim $(find . -name "*.py" -newer requirements.txt | fzf)
```

### Recommended ~/.bashrc / ~/.zshrc Config

```bash
# Use fd (faster find) as fzf's source if available
export FZF_DEFAULT_COMMAND='fd --type f --hidden --follow --exclude .git'
export FZF_CTRL_T_COMMAND="$FZF_DEFAULT_COMMAND"

# Show file preview in Ctrl+t search
export FZF_CTRL_T_OPTS="--preview 'bat --color=always --line-range :50 {}'"

# Show directory tree preview in Alt+c search
export FZF_ALT_C_OPTS="--preview 'tree -C {} | head -50'"
```

---

## B.17 GNU parallel — Parallel Execution

`GNU parallel` runs shell commands in parallel across CPU cores — far more flexible than `xargs -P` for complex workloads like batch preprocessing, hyperparameter sweeps, or running the same script over many files.

```bash
# Install
sudo apt install parallel       # Ubuntu/Debian
brew install parallel           # macOS

# Basic syntax: parallel [options] command ::: arguments
parallel echo ::: A B C D
# → A  B  C  D  (all at once, order may vary)

# Run a Python script on every CSV file using all CPU cores
parallel python process.py {} ::: data/*.csv

# Limit to N simultaneous jobs
parallel -j4 python process.py {} ::: data/*.csv

# Read arguments from a file (one per line)
parallel -j4 python process.py {} :::: file_list.txt

# Pass multiple arguments
parallel python train.py --lr {1} --batch {2} ::: 0.001 0.0001 ::: 16 32

# Log which jobs succeeded / failed
parallel --joblog parallel.log python train.py {} ::: configs/*.yaml

# Show progress bar
parallel --progress python process.py {} ::: data/*.json

# Retry failed jobs up to 3 times
parallel --retries 3 python process.py {} ::: data/*.json

# Resume from a previous log (skip already-completed jobs)
parallel --resume --joblog parallel.log python train.py {} ::: configs/*.yaml
```

### ML Batch Processing Example

```bash
# Preprocess 1000 audio files using 8 cores
ls raw_audio/*.wav | parallel -j8 python preprocess.py --input {} --output processed/{/.}.pt

# Hyperparameter sweep: all combinations of LR × batch size
parallel -j4 --joblog sweep.log \
    python train.py --lr {1} --batch-size {2} --output results/lr{1}_bs{2}/ \
    ::: 1e-3 1e-4 1e-5 \
    ::: 16 32 64

# Evaluate a saved model on multiple test sets
parallel python eval.py --checkpoint best.pt --data {} ::: test_sets/*.json \
    | tee eval_results.txt
```

### GNU parallel vs xargs

| Feature | `xargs -P` | `GNU parallel` |
|---|---|---|
| Multiple argument sources | ✗ | ✓ |
| Job logging | ✗ | ✓ |
| Resume failed runs | ✗ | ✓ |
| Progress display | ✗ | ✓ |
| Retry on failure | ✗ | ✓ |
| Argument replacement `{}` | Limited | Full |

---

*Last updated: April 2026*
