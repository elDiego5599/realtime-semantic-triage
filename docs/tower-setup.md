# Tower setup (handoff for the Windows tower session)

This note is for the Claude Code session that runs on the Windows tower. Read `MASTER_PLAN.md` first, then this file. It explains what the tower is for and what to set up there, in order, with a check for each step.

## Why the tower

Phase 1 (recording, labeled dataset, offline dedup baseline) runs on the MacBook. From phase 2 on, the heavy parts run on the tower:

- Qdrant in Docker (dense plus sparse collection, payload indexes).
- bge-m3 embeddings (dense and sparse) and bge-reranker-v2-m3 through FlagEmbedding on the GPU.
- Ollama with an 8B instruct model for phase 4 diagnoses.
- The FastAPI backend, the only service exposed to the LAN, behind an API key.

The MacBook keeps Streamlit, evaluation and development, and reaches the tower over the LAN.

Setup can start now, in parallel with the rest of phase 1. Nothing in phase 1 depends on it.

## Ground rules

- Same rules as every session: `MASTER_PLAN.md` is the authority, work on branches, never on `main`, no push, merge or PR without Diego's approval. Commits in English, imperative, short, without a Co-Authored-By line.
- Code tasks are delegated to an implementer (Copilot CLI with a fresh `copilot -p` per brief, or DeepSeek through opencode when its free limit allows). This setup is mostly machine configuration, so the session runs the checks itself. Anything that needs a password, an administrator prompt or `sudo` (driver install, Docker Desktop install, firewall rules) is done by Diego.
- Nothing private goes into the repository: no IP addresses, host names, user names, keys or tokens. The repository is public.
- Pin every image and tool version you install, and record the versions in the `MASTER_PLAN.md` session log.

## Steps

### 1. GPU visible from WSL2

The RTX 5070 is Blackwell (`sm_120`). It needs CUDA 12.8 or newer.

- Install or update the NVIDIA driver on Windows only. Do not install a Linux NVIDIA driver inside WSL2.
- Use an Ubuntu distribution under WSL2.
- Check, inside WSL2: `nvidia-smi` lists the RTX 5070 and reports CUDA version 12.8 or newer.

### 2. Repository and Python inside WSL2

- Clone the repository into the WSL2 file system (for example `~/realtime-semantic-triage`), never under `/mnt/c`, which is slow and breaks file watching.
- Install `uv`, then run `uv sync`.
- Check: `uv run pytest` passes and `uv run ruff check .` is clean.

### 3. PyTorch with CUDA 12.8

Verify the GPU from Python in a throwaway environment. Do not add `torch` to `pyproject.toml` yet: that belongs to the phase 2 task that introduces the embedder.

```bash
uv venv /tmp/torchcheck
VIRTUAL_ENV=/tmp/torchcheck uv pip install torch --index-url https://download.pytorch.org/whl/cu128
/tmp/torchcheck/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
```

Check: `True` and capability `(12, 0)`.

### 4. Docker and Qdrant

- Docker Desktop with the WSL2 backend, or Docker Engine inside WSL2. Pick one and record which.
- Run Qdrant with a pinned tag (the current stable release at setup time), a named volume, and ports bound to localhost only:

```bash
docker run -d --name qdrant --restart unless-stopped \
  -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant:<pinned-tag>
```

- Never bind mount a Windows path (`C:\` or `/mnt/c`) as storage. Use the named volume.
- Check: `curl -s 127.0.0.1:6333/readyz` answers, and the container survives a restart of Docker.

### 5. Ollama

- Install Ollama (inside WSL2 or on Windows; record which) and keep it on `127.0.0.1:11434`.
- Check: `curl -s 127.0.0.1:11434/api/version` answers and `ollama ps` shows the GPU in use after a short test prompt.
- Pulling the 8B instruct model can wait until phase 4. Leave the model name configurable.

### 6. Network

- Nothing is exposed to the LAN yet. Qdrant and Ollama stay on 127.0.0.1.
- When the FastAPI backend exists (phase 2 or 3), expose only its port, behind an API key. WSL2 needs either mirrored networking or a port proxy for that, plus a Windows firewall rule limited to the private network. Diego applies those.

### 7. Optional: drive the tower from the MacBook

Diego can set up SSH from the MacBook into WSL2 (his own keys) so that MacBook sessions can start work on the tower. Not required: a local Claude Code session on the tower works too.

## Done criteria

- `nvidia-smi` in WSL2 shows the RTX 5070 with CUDA 12.8 or newer.
- `uv run pytest` and `uv run ruff check .` pass in the WSL2 clone.
- PyTorch cu128 reports `cuda.is_available()` and capability `(12, 0)`.
- Qdrant answers on `127.0.0.1:6333` with a pinned tag and a named volume.
- Ollama answers on `127.0.0.1:11434`.
- A session log entry in `MASTER_PLAN.md` records the driver, CUDA, Docker, Qdrant tag, Ollama and PyTorch versions, on a branch, committed for Diego to review.
