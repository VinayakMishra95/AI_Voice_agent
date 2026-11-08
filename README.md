# Mermaid Voice Agent (100% Local)

A 100% local, Apple Silicon voice assistant with a calm, pleasant voice powered by [MLX-Audio](https://github.com/Blaizzy/mlx-audio), [Kokoro-82M](https://huggingface.co/mlx-community/Kokoro-82M-bf16), Whisper STT, and `Qwen2.5-1.5B-Instruct`.

## 1. Hands-Free Voice Agent (Speak & Listen)

Open **Terminal** in `/Users/vinayak/Projects/Mermaid_Voice_Agent` and run:

```bash
./.venv/bin/python mermaid_agent.py --voice af_heart --ethereal
```

* **First run:** Downloads the local Whisper STT (`whisper-small-asr-fp16`) and `Qwen2.5-1.5B-Instruct` chat model once to your Hugging Face cache. Every run after that works **100% offline**.
* **Permissions:** macOS will prompt once to grant **Microphone** access to Terminal.
* **How it works:** Calibrates your room's ambient noise for ~0.6s, listens automatically when you speak, pauses while Mermaid replies, and exits when you say **"goodbye"** (or press `Ctrl+C`).

## 2. Text-to-Speech Only (Typing Mode)

Audition the five pleasant voices:
```bash
./.venv/bin/python mermaid_tts.py --audition --ethereal
```

Interactive typing mode:
```bash
./.venv/bin/python mermaid_tts.py --voice af_heart --ethereal
```

## Recommended Voices
* `af_heart` — Warm, gentle, expressive American English (default)
* `af_bella` — Smooth, clear, soothing American English
* `af_sky` — Light, airy, youthful American English
* `af_nicole` — Soft, calm, intimate American English
* `bf_emma` — Melodic, poised British English
# AI_Voice_agent
