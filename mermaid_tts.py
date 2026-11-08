#!/usr/bin/env python3
"""
Mermaid Voice Agent — 100% Local Text-to-Speech for Apple Silicon (MLX-Audio + Kokoro-82M).
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy.signal import butter, lfilter

DEFAULT_MODEL_ID = "mlx-community/Kokoro-82M-bf16"
LOCAL_MODEL_DIR = Path(__file__).parent / "models" / "Kokoro-82M-bf16"
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Pleasant, soft, melodic female voices in Kokoro-82M
MERMAID_VOICES = {
    "af_heart": "Warm, gentle, expressive American English (default)",
    "af_bella": "Smooth, clear, soothing American English",
    "af_sky": "Light, airy, youthful American English",
    "af_nicole": "Soft, calm, intimate American English",
    "bf_emma": "Melodic, poised British English",
}


def lang_code_for_voice(voice: str) -> str:
    """Kokoro uses 'b' for British voices (bf_*/bm_*) and 'a' for American voices (af_*/am_*)."""
    return "b" if voice.startswith(("bf_", "bm_")) else "a"


def apply_ethereal_polish(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    """
    Add a subtle, pleasant studio warmth and soft shimmer tail without muddying speech.
    Pure NumPy/SciPy so it works 100% locally with zero extra dependencies.
    """
    if audio.size == 0:
        return audio

    audio = audio.astype(np.float32, copy=False)

    # Gentle high-shelf softening above 7.5 kHz to remove any digital harshness
    b, a = butter(2, 7500 / (0.5 * sample_rate), btype="low")
    warm = lfilter(b, a, audio).astype(np.float32)

    # Very light multi-tap shimmer tail (low wet mix so diction stays crisp)
    taps_ms = [(38, 0.09), (79, 0.055), (131, 0.03)]
    max_delay = int(max(ms for ms, _ in taps_ms) * sample_rate / 1000) + 1
    padded = np.pad(warm, (0, max_delay))
    wet = np.zeros_like(padded)
    for ms, gain in taps_ms:
        delay = int(ms * sample_rate / 1000)
        wet[delay : delay + len(warm)] += warm * gain

    mixed = padded * 0.94 + wet
    peak = float(np.max(np.abs(mixed))) if mixed.size else 1.0
    if peak > 0.98:
        mixed = mixed * (0.98 / peak)
    return mixed.astype(np.float32)


def load_kokoro(model_source: str):
    from mlx_audio.tts.utils import load_model

    print(f"Loading local TTS model from: {model_source}")
    model = load_model(model_source)
    return model


def synthesize(
    model,
    text: str,
    voice: str = "af_heart",
    speed: float = 0.94,
    ethereal: bool = False,
) -> tuple[np.ndarray, int]:
    """Synthesize speech into a single NumPy float32 waveform."""
    chunks = []
    sample_rate = getattr(model, "sample_rate", 24000)
    lang_code = lang_code_for_voice(voice)

    for result in model.generate(
        text=text.strip(),
        voice=voice,
        speed=speed,
        lang_code=lang_code,
        split_pattern=r"\n+",
    ):
        arr = np.asarray(result.audio, dtype=np.float32).squeeze()
        if arr.size > 0:
            chunks.append(arr)
            sample_rate = result.sample_rate

    if not chunks:
        return np.zeros(0, dtype=np.float32), sample_rate

    # Short 120ms breath pause between multi-line segments
    if len(chunks) > 1:
        pause = np.zeros(int(sample_rate * 0.12), dtype=np.float32)
        joined = []
        for idx, chunk in enumerate(chunks):
            joined.append(chunk)
            if idx < len(chunks) - 1:
                joined.append(pause)
        audio = np.concatenate(joined)
    else:
        audio = chunks[0]

    if ethereal:
        audio = apply_ethereal_polish(audio, sample_rate)

    return audio, sample_rate


def play_audio(audio: np.ndarray, sample_rate: int) -> None:
    if audio.size == 0:
        return
    sd.play(audio, samplerate=sample_rate, blocking=True)


def sanitize_filename(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")
    return cleaned or "mermaid_speech"


def run_audition(model, speed: float, ethereal: bool) -> None:
    sample_line = (
        "Hello there. Close your eyes for a moment and listen to the calm, "
        "steady rhythm of the tide."
    )
    print("\nAuditioning pleasant Kokoro voices:\n")
    for voice, desc in MERMAID_VOICES.items():
        print(f"  -> {voice}: {desc}")
        audio, sr = synthesize(model, sample_line, voice=voice, speed=speed, ethereal=ethereal)
        play_audio(audio, sr)
    print("\nAudition complete. Pass --voice <name> to choose your favorite.\n")


def run_interactive(model, voice: str, speed: float, ethereal: bool) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    last_audio: Optional[np.ndarray] = None
    last_sr: int = 24000

    print("\nMermaid Voice Agent (100% Local TTS)")
    print(f"Voice: {voice} | Speed: {speed:.2f}x | Ethereal polish: {'on' if ethereal else 'off'}")
    print("Type any text to speak it aloud. Commands:")
    print("  /voices              List recommended pleasant voices")
    print("  /voice <name>        Switch voice (e.g. /voice af_sky)")
    print("  /speed <num>         Set speed (e.g. /speed 0.92)")
    print("  /ethereal on|off     Toggle subtle warmth & shimmer")
    print("  /save <filename>     Save the last spoken clip to outputs/")
    print("  /quit                Exit\n")

    while True:
        try:
            line = input("Speak > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not line:
            continue

        if line in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye.")
            break

        if line == "/voices":
            for v, desc in MERMAID_VOICES.items():
                marker = "*" if v == voice else " "
                print(f" {marker} {v:10s} — {desc}")
            continue

        if line.startswith("/voice "):
            new_voice = line.split(maxsplit=1)[1].strip()
            voice = new_voice
            print(f"Switched voice to: {voice}")
            continue

        if line.startswith("/speed "):
            try:
                speed = float(line.split(maxsplit=1)[1].strip())
                print(f"Speed set to: {speed:.2f}x")
            except ValueError:
                print("Please provide a valid number, e.g. /speed 0.92")
            continue

        if line.startswith("/ethereal "):
            arg = line.split(maxsplit=1)[1].strip().lower()
            ethereal = arg in {"on", "true", "1", "yes"}
            print(f"Ethereal polish: {'on' if ethereal else 'off'}")
            continue

        if line.startswith("/save "):
            if last_audio is None or last_audio.size == 0:
                print("Nothing synthesized yet to save.")
                continue
            fname = sanitize_filename(line.split(maxsplit=1)[1].strip())
            if not fname.endswith(".wav"):
                fname += ".wav"
            out_path = OUTPUT_DIR / fname
            sf.write(out_path, last_audio, last_sr)
            print(f"Saved to {out_path}")
            continue

        audio, sr = synthesize(model, line, voice=voice, speed=speed, ethereal=ethereal)
        last_audio, last_sr = audio, sr
        play_audio(audio, sr)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="100% local, pleasant Mermaid-style Text-to-Speech on Apple Silicon."
    )
    parser.add_argument("--text", type=str, help="Text to speak")
    parser.add_argument("--file", type=Path, help="Read text from a UTF-8 file")
    parser.add_argument("--voice", type=str, default="af_heart", help="Kokoro voice preset")
    parser.add_argument("--speed", type=float, default=0.94, help="Speaking speed (default: 0.94)")
    parser.add_argument(
        "--ethereal",
        action="store_true",
        help="Apply subtle warmth and soft shimmer tail",
    )
    parser.add_argument("--out", type=Path, help="Save output WAV to this path")
    parser.add_argument("--no-play", action="store_true", help="Do not play audio on speakers")
    parser.add_argument(
        "--audition",
        action="store_true",
        help="Play a short sample of each recommended pleasant voice",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else DEFAULT_MODEL_ID,
        help="Local model directory or Hugging Face repo ID",
    )
    args = parser.parse_args()

    model = load_kokoro(args.model)

    if args.audition:
        run_audition(model, speed=args.speed, ethereal=args.ethereal)
        return 0

    text = args.text
    if args.file is not None:
        text = args.file.read_text(encoding="utf-8")

    if text is not None:
        audio, sr = synthesize(
            model,
            text=text,
            voice=args.voice,
            speed=args.speed,
            ethereal=args.ethereal,
        )
        if args.out is not None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(args.out, audio, sr)
            print(f"Saved WAV to: {args.out}")
        if not args.no_play:
            play_audio(audio, sr)
        return 0

    run_interactive(model, voice=args.voice, speed=args.speed, ethereal=args.ethereal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
