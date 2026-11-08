# Mermaid Voice Agent

A 100% local, pleasant-voiced conversational AI assistant with a hands-free browser UI, containerized with **Docker** and orchestrated with **Kubernetes (Minikube)**.

## Architecture Overview

```
Browser (static/index.html)
  │  Web Audio API + adaptive noise-floor VAD + per-tab X-Session-Id
  ▼  POST /api/talk (16-bit PCM WAV)
Kubernetes Service (mermaid-voice-agent:80 -> Pod:8000)
  │
  ├──► FastAPI + Uvicorn (app/main.py)
  │      ├── 1. Resample audio to 16 kHz (scipy)
  │      ├── 2. Speech-to-Text: faster-whisper (small.en, int8)
  │      ├── 3. Session Memory: TTL/LRU SessionStore (per X-Session-Id)
  │      ├── 4. LLM Request ──► ClusterIP Service (ollama:11434/v1)
  │      │                         └── Ollama Pod (qwen2.5:1.5b)
  │      └── 5. Text-to-Speech: kokoro-onnx (Kokoro-82M, af_heart + optional DSP warmth)
  │
  └──◄ Returns JSON { user_text, reply_text, audio_b64 (16-bit PCM WAV) }
```

## Tech Stack

| Layer | Production (Docker / Kubernetes) | Local macOS Dev (Apple Silicon) |
| :--- | :--- | :--- |
| **Frontend** | Vanilla HTML/CSS/JS (`static/index.html`), Web Audio API VAD | Same `static/index.html` |
| **API Server** | FastAPI + Uvicorn (`app/main.py`) | `wsgiref` (`mermaid_web.py`) / CLI (`mermaid_agent.py`) |
| **Speech-to-Text (STT)** | `faster-whisper` (`small.en`, `int8`, CTranslate2) | `mlx-audio` (`whisper-small-asr-fp16` on Metal) |
| **LLM Brain** | `Ollama` (`qwen2.5:1.5b`) via OpenAI-compatible HTTP API | `Qwen/Qwen2.5-1.5B-Instruct` via `transformers` |
| **Text-to-Speech (TTS)** | `kokoro-onnx` (Kokoro-82M ONNX Runtime, `af_heart` voice) | `mlx-audio` (`Kokoro-82M-bf16` on Metal) |
| **Audio Processing** | `soundfile`, `scipy` (resampling & ethereal low-pass/shimmer) | `soundfile`, `scipy`, `afplay` |
| **Containerization** | Docker (`python:3.12-slim`) | N/A |
| **Orchestration** | Kubernetes (`k8s/` manifests on Minikube) | N/A |

## What Docker Does (`Dockerfile` & `.dockerignore`)

The [`Dockerfile`](Dockerfile) packages the voice API into a portable, reproducible Linux container:

1. **Base Image (`python:3.12-slim`):** Keeps the image lightweight while providing Python 3.12.
2. **System Audio Dependencies:** Installs `libsndfile1`, `ffmpeg`, and `espeak-ng` required by `soundfile` and `kokoro-onnx` phoneme processing.
3. **Production Dependencies:** Installs only the Linux production requirements from [`requirements.prod.txt`](requirements.prod.txt) (`fastapi`, `uvicorn`, `kokoro-onnx`, `faster-whisper`, `httpx`, `scipy`, `soundfile`).
4. **Slim Build Context (`.dockerignore`):** Excludes `.venv/`, `models/`, `outputs/`, and macOS-only `mlx` scripts so the container stays clean and builds fast.
5. **Externalized Model Cache (`/models`):** Instead of baking gigabytes of weights into the Docker image, the container mounts `/models` from a Kubernetes `PersistentVolumeClaim` so weights are downloaded once and reused across restarts.

## How Kubernetes & Orchestration Work (`k8s/`)

All Kubernetes manifests live in [`k8s/`](k8s/) and deploy two decoupled microservices into an isolated `mermaid` namespace:

1. **Namespace Isolation ([`00-namespace.yaml`](k8s/00-namespace.yaml)):** Creates the `mermaid` namespace so all services, PVCs, and configs stay cleanly separated from other cluster workloads.
2. **Persistent Storage ([`01-pvc.yaml`](k8s/01-pvc.yaml)):**
   * `mermaid-models-pvc` (5 Gi) mounted at `/models` stores `kokoro-v1.0.onnx`, `voices-v1.0.bin`, and `faster-whisper` weights.
   * `ollama-models-pvc` (8 Gi) mounted at `/root/.ollama` persists `qwen2.5:1.5b`.
3. **12-Factor Configuration ([`02-configmap.yaml`](k8s/02-configmap.yaml)):** Injects `STT_MODEL`, `STT_COMPUTE_TYPE`, `LLM_BASE_URL`, `LLM_MODEL`, and `MODEL_CACHE_DIR` into the app container via `envFrom`, allowing environment-specific swaps without rebuilding the image.
4. **Decoupled LLM Microservice ([`03-ollama.yaml`](k8s/03-ollama.yaml)):** Runs `ollama/ollama:latest` as its own `Deployment` behind a `ClusterIP` Service (`ollama.mermaid.svc.cluster.local:11434`). This separates LLM compute from the voice gateway so the LLM tier can be scaled or moved to GPU nodes independently.
5. **Voice Gateway & Health Probes ([`04-app.yaml`](k8s/04-app.yaml)):**
   * **Readiness Probe (`GET /readyz`):** Returns `503` while `kokoro-onnx` and `faster-whisper` are loading into memory, and `200` once ready—ensuring Kubernetes never routes user traffic to a warming pod.
   * **Liveness Probe (`GET /healthz`):** Automatically restarts the container if the Uvicorn process becomes unresponsive.
   * **Resource Requests & Limits:** Reserves `1 CPU / 1.5 Gi RAM` and caps at `4 CPU / 4 Gi RAM` to prevent node OOM-kills.
   * **Service (`NodePort 30080`):** Exposes port `80` -> container port `8000` for local Minikube access.
6. **External Ingress ([`05-ingress.yaml`](k8s/05-ingress.yaml)):** Defines HTTP routing (`mermaid.local`) with a `120s` proxy read timeout for inference, ready to pair with `cert-manager` TLS for cloud deployment.

## Quick Start: Minikube (Local Production)

```bash
# 1. Start Colima & Minikube
colima start --cpu 4 --memory 8
minikube start --driver=docker --cpus=4 --memory=7168

# 2. Build the Docker image and load it into Minikube
docker build -t mermaid-voice-agent:1.0.0 .
minikube image load mermaid-voice-agent:1.0.0

# 3. Deploy all Kubernetes manifests
kubectl apply -f k8s/
kubectl -n mermaid exec deployment/ollama -- ollama pull qwen2.5:1.5b

# 4. Forward the service and open http://127.0.0.1:8080
kubectl -n mermaid port-forward svc/mermaid-voice-agent 8080:80
```

## Quick Start: Local macOS (Apple Silicon)

```bash
cd /Users/vinayak/Projects/Mermaid_Voice_Agent
./.venv/bin/python mermaid_web.py
```

## Cloud Production Roadmap

* **TLS / HTTPS:** Terminate Let's Encrypt TLS at the Kubernetes `Ingress` (required by browsers for `getUserMedia` microphone access on public domains).
* **Shared Session State:** Move `SessionStore` from in-memory to **Redis** and use `ReadOnlyMany` model storage so multiple `mermaid-voice-agent` replicas can scale horizontally behind a `HorizontalPodAutoscaler (HPA)`.
* **GPU & Streaming:** Move the LLM tier to a GPU node pool (`vLLM` or Ollama GPU) and stream sentence-by-sentence TTS over WebSockets for sub-second response latency.
