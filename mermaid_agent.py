#!/usr/bin/env python3
"""
Mermaid Voice Agent — 100% Local Hands-Free Voice Assistant for Apple Silicon.

Pipeline (all running locally on your Mac):
  Mic (adaptive VAD) -> Whisper STT (MLX-Audio) -> Qwen2.5 LLM (MPS) -> Kokoro-82M TTS (MLX-Audio)
"""

from __future__ import annotations

import argparse
import base64
import collections
import io
import json
import math
import re
import sys
import threading
import time
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import List, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import AutoModelForCausalLM, AutoTokenizer

from mermaid_tts import (
    DEFAULT_MODEL_ID as DEFAULT_TTS_MODEL,
    LOCAL_MODEL_DIR as LOCAL_TTS_DIR,
    MERMAID_VOICES,
    load_kokoro,
    play_audio,
    synthesize,
)

DEFAULT_STT_MODEL = "mlx-community/whisper-small-asr-fp16"
DEFAULT_LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)

SYSTEM_PROMPT = (
    "You are Mermaid, a calm, warm, and genuinely pleasant voice companion running "
    "completely locally on the user's Mac. Speak in natural, gentle, everyday English "
    "using short conversational replies (1 to 3 sentences unless asked for detail). "
    "Never use markdown, bullet points, asterisks, emojis, or URLs because your words "
    "are spoken aloud."
)

EXIT_PHRASES = {
    "goodbye",
    "good bye",
    "bye mermaid",
    "stop listening",
    "exit",
    "quit",
}


def load_stt(model_id: str):
    from mlx_audio.stt.utils import load_model as load_stt_model

    print(f"Loading local STT model: {model_id}")
    return load_stt_model(model_id)


def load_llm(model_id: str):
    print(f"Loading local LLM: {model_id}", flush=True)
    # Keep PyTorch on CPU so it never contends with MLX on Apple Silicon's Metal GPU
    device = "cpu"
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
    ).to(device)
    model.eval()
    return tokenizer, model, device


def calibrate_noise_floor(seconds: float = 0.6) -> float:
    """Measure ambient room noise RMS so VAD adapts to your mic and environment."""
    frames = max(1, int(seconds * SAMPLE_RATE / FRAME_SAMPLES))
    rms_values: List[float] = []

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
    ) as stream:
        for _ in range(frames):
            data, _ = stream.read(FRAME_SAMPLES)
            rms = float(np.sqrt(np.mean(np.square(data))))
            rms_values.append(rms)

    baseline = float(np.median(rms_values)) if rms_values else 0.005
    # Set trigger threshold safely above room hum, with sensible floor/ceiling
    threshold = max(0.010, min(0.06, baseline * 3.0 + 0.006))
    return threshold


def record_utterance(
    threshold: float,
    silence_seconds: float = 1.1,
    max_seconds: float = 20.0,
) -> np.ndarray:
    """
    Hands-free recording: waits for speech above threshold, then records until
    `silence_seconds` of quiet after speech ends.
    """
    pre_roll_frames = int(0.30 * 1000 / FRAME_MS)  # 300ms before trigger
    silence_limit = int(silence_seconds * 1000 / FRAME_MS)
    max_frames = int(max_seconds * 1000 / FRAME_MS)
    min_speech_frames = int(0.25 * 1000 / FRAME_MS)

    ring: collections.deque[np.ndarray] = collections.deque(maxlen=pre_roll_frames)
    recorded: List[np.ndarray] = []
    triggered = False
    speech_frames = 0
    silent_frames = 0

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=FRAME_SAMPLES,
    ) as stream:
        while True:
            data, _ = stream.read(FRAME_SAMPLES)
            frame = data[:, 0].copy()
            rms = float(np.sqrt(np.mean(np.square(frame))))

            if not triggered:
                ring.append(frame)
                if rms >= threshold:
                    speech_frames += 1
                    if speech_frames >= 2:
                        triggered = True
                        recorded.extend(ring)
                        silent_frames = 0
                else:
                    speech_frames = 0
            else:
                recorded.append(frame)
                if rms >= threshold * 0.75:
                    silent_frames = 0
                    speech_frames += 1
                else:
                    silent_frames += 1

                if silent_frames >= silence_limit or len(recorded) >= max_frames:
                    break

    if speech_frames < min_speech_frames or not recorded:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(recorded).astype(np.float32)


def transcribe(stt_model, audio: np.ndarray) -> str:
    if audio.size < int(SAMPLE_RATE * 0.35):
        return ""
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 0.004:
        return ""
    peak = float(np.max(np.abs(audio)))
    if 0 < peak < 0.25:
        audio = (audio / peak) * 0.5
    result = stt_model.generate(
        audio,
        language="en",
        verbose=False,
        condition_on_previous_text=False,
        temperature=0.0,
        no_speech_threshold=0.5,
        compression_ratio_threshold=2.0,
    )
    text = (result.text or "").strip()
    # Filter common Whisper silence/hallucination artifacts
    cleaned = re.sub(r"[\[\(].*?[\]\)]", "", text).strip()
    if cleaned.lower() in {"", "you", "thank you.", "thanks for watching.", "bye."}:
        return ""
    return cleaned


def generate_reply(
    tokenizer,
    llm_model,
    device: str,
    history: List[dict],
    user_text: str,
    max_new_tokens: int = 80,
) -> Tuple[str, List[dict]]:
    print(f"You     > {user_text}", flush=True)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_text}
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

    with torch.no_grad():
        output_ids = llm_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1] :]
    reply = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    # Strip any stray markdown formatting so TTS reads cleanly
    reply = re.sub(r"[*_`#]+", "", reply).strip()
    print(f"Mermaid > {reply}", flush=True)

    updated_history = (
        history
        + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
    )[-16:]  # Keep last 8 turns
    return reply, updated_history


def main() -> int:
    parser = argparse.ArgumentParser(
        description="100% local hands-free Mermaid Voice Agent on Apple Silicon."
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="af_heart",
        choices=list(MERMAID_VOICES.keys()),
        help="Kokoro voice preset (default: af_heart)",
    )
    parser.add_argument("--speed", type=float, default=0.94, help="TTS speed (default: 0.94)")
    parser.add_argument(
        "--ethereal",
        action="store_true",
        help="Apply subtle warmth and soft shimmer tail to Mermaid's voice",
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default=DEFAULT_STT_MODEL,
        help=f"Whisper MLX model (default: {DEFAULT_STT_MODEL})",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=DEFAULT_LLM_MODEL,
        help=f"Local Hugging Face chat model (default: {DEFAULT_LLM_MODEL})",
    )
    parser.add_argument(
        "--tts-model",
        type=str,
        default=str(LOCAL_TTS_DIR) if LOCAL_TTS_DIR.exists() else DEFAULT_TTS_MODEL,
        help="Local Kokoro TTS model path or HF repo ID",
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=1.1,
        help="Seconds of silence before ending your turn (default: 1.1)",
    )
    args = parser.parse_args()

    tts_model = load_kokoro(args.tts_model)
    stt_model = load_stt(args.stt_model)
    tokenizer, llm_model, device = load_llm(args.llm_model)

    print("\nCalibrating microphone for ambient noise (stay quiet for a moment)...")
    threshold = calibrate_noise_floor()
    print(f"Ready! (VAD threshold: {threshold:.4f}, voice: {args.voice})")
    print("Just speak naturally. Say 'goodbye' or press Ctrl+C to stop.\n")

    greeting = "Hello! I'm listening whenever you're ready."
    print(f"Mermaid > {greeting}")
    audio_out, sr = synthesize(
        tts_model, greeting, voice=args.voice, speed=args.speed, ethereal=args.ethereal
    )
    play_audio(audio_out, sr)

    history: List[dict] = []

    try:
        while True:
            print("\nListening...")
            utterance = record_utterance(threshold=threshold, silence_seconds=args.silence)
            if utterance.size == 0:
                continue

            t0 = time.time()
            user_text = transcribe(stt_model, utterance)
            if not user_text:
                continue

            print(f"You     > {user_text}")

            normalized = re.sub(r"[^\w\s]", "", user_text.lower()).strip()
            if normalized in EXIT_PHRASES:
                farewell = "Goodbye for now. Talk to you soon."
                print(f"Mermaid > {farewell}")
                audio_out, sr = synthesize(
                    tts_model,
                    farewell,
                    voice=args.voice,
                    speed=args.speed,
                    ethereal=args.ethereal,
                )
                play_audio(audio_out, sr)
                break

            reply, history = generate_reply(
                tokenizer, llm_model, device, history, user_text
            )
            if not reply:
                continue

            elapsed = time.time() - t0
            print(f"Mermaid > {reply}  ({elapsed:.1f}s)")
            audio_out, sr = synthesize(
                tts_model,
                reply,
                voice=args.voice,
                speed=args.speed,
                ethereal=args.ethereal,
            )
            play_audio(audio_out, sr)

    except KeyboardInterrupt:
        print("\nStopped listening. Goodbye!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
