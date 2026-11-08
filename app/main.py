#!/usr/bin/env python3
"""
Mermaid Voice Agent — Production FastAPI Backend (Linux / Kubernetes / Minikube).

Two-tier Redis memory:
  1. Active conversation turns (`mermaid:session:<id>`) — cleared on New Chat / Clear.
  2. Persistent long-term memory (`mermaid:memory:<id>`) — survives Clear & restarts,
     extracted asynchronously via the local LLM so past facts & conversations remain
     searchable and recallable without adding reply latency.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import re
import time
import urllib.request
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import httpx
import numpy as np
import redis.asyncio as aioredis
import soundfile as sf
from fastapi import FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from scipy.signal import butter, lfilter, resample_poly

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("mermaid")

BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_HTML_PATH = BASE_DIR / "static" / "index.html"
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", "/models"))

AGENT_NAME = os.getenv("AGENT_NAME", "Mermaid")
KOKORO_ONNX_URL = os.getenv(
    "KOKORO_ONNX_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx",
)
KOKORO_VOICES_URL = os.getenv(
    "KOKORO_VOICES_URL",
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin",
)

STT_MODEL_NAME = os.getenv("STT_MODEL", "small.en")
STT_COMPUTE_TYPE = os.getenv("STT_COMPUTE_TYPE", "int8")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://ollama:11434/v1").rstrip("/")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
LLM_API_KEY = os.getenv("LLM_API_KEY", "local-ollama")
REDIS_URL = os.getenv("REDIS_URL", "")
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))
SESSION_TTL_SECONDS = int(os.getenv("SESSION_TTL_SECONDS", "604800"))   # 7 days
MEMORY_TTL_SECONDS = int(os.getenv("MEMORY_TTL_SECONDS", "2592000"))    # 30 days

MERMAID_VOICES = {
    "af_heart": "Warm, gentle, expressive American English (default)",
    "af_bella": "Smooth, clear, soothing American English",
    "af_sky": "Light, airy, youthful American English",
    "af_nicole": "Soft, calm, intimate American English",
    "bf_emma": "Melodic, poised British English",
}

BASE_SYSTEM_PROMPT = (
    f"You are {AGENT_NAME}, a calm, warm, and genuinely intelligent voice companion. "
    "You have persistent long-term memory across conversations. Always use both the "
    "long-term memory notes and the current conversation history to answer questions "
    "accurately. Speak in natural, gentle, everyday English using short conversational "
    "replies (1 to 3 sentences unless asked for detail). Never use markdown, bullet "
    "points, asterisks, emojis, or URLs because your words are spoken aloud."
)


def apply_ethereal_polish(audio: np.ndarray, sample_rate: int = 24000) -> np.ndarray:
    if audio.size == 0:
        return audio
    audio = audio.astype(np.float32, copy=False)
    b, a = butter(2, 7500 / (0.5 * sample_rate), btype="low")
    warm = lfilter(b, a, audio).astype(np.float32)
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


def ensure_file(path: Path, url: str) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    logger.info("Downloading %s -> %s", url, path)
    urllib.request.urlretrieve(url, tmp_path)
    tmp_path.replace(path)
    return path


class SessionStore:
    def __init__(self, max_sessions: int, ttl_seconds: int):
        self.max_sessions = max_sessions
        self.ttl_seconds = ttl_seconds
        self._sessions: OrderedDict[str, tuple[float, List[dict]]] = OrderedDict()
        self._memories: OrderedDict[str, str] = OrderedDict()
        self.redis: Optional[aioredis.Redis] = None

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        if not REDIS_URL:
            return None
        if self.redis is not None:
            return self.redis
        try:
            client = aioredis.from_url(REDIS_URL, decode_responses=True)
            await client.ping()
            self.redis = client
            logger.info("Connected to Redis session & memory store.")
            return self.redis
        except Exception as exc:
            logger.warning("Redis connect failed (%s); using memory fallback.", exc)
            self.redis = None
            return None

    async def get_history(self, session_id: str) -> List[dict]:
        r = await self._get_redis()
        if r is not None:
            try:
                raw = await r.get(f"mermaid:session:{session_id}")
                return json.loads(raw) if raw else []
            except Exception as exc:
                logger.warning("Redis get_history failed: %s", exc)
                self.redis = None

        now = time.time()
        self._evict_expired(now)
        if session_id in self._sessions:
            _, history = self._sessions.pop(session_id)
            self._sessions[session_id] = (now, history)
            return list(history)
        return []

    async def set_history(self, session_id: str, history: List[dict]) -> None:
        trimmed = history[-20:]
        r = await self._get_redis()
        if r is not None:
            try:
                await r.set(
                    f"mermaid:session:{session_id}",
                    json.dumps(trimmed),
                    ex=self.ttl_seconds,
                )
                return
            except Exception as exc:
                logger.warning("Redis set_history failed: %s", exc)
                self.redis = None

        now = time.time()
        self._evict_expired(now)
        if session_id in self._sessions:
            self._sessions.pop(session_id)
        elif len(self._sessions) >= self.max_sessions:
            self._sessions.popitem(last=False)
        self._sessions[session_id] = (now, trimmed)

    async def clear_history(self, session_id: str) -> None:
        """Clears only the active UI thread; preserves long-term memory."""
        r = await self._get_redis()
        if r is not None:
            try:
                await r.delete(f"mermaid:session:{session_id}")
            except Exception as exc:
                logger.warning("Redis clear_history failed: %s", exc)
                self.redis = None
        self._sessions.pop(session_id, None)

    async def get_memory(self, session_id: str) -> str:
        r = await self._get_redis()
        if r is not None:
            try:
                return (await r.get(f"mermaid:memory:{session_id}")) or ""
            except Exception as exc:
                logger.warning("Redis get_memory failed: %s", exc)
                self.redis = None
        return self._memories.get(session_id, "")

    async def set_memory(self, session_id: str, memory_text: str) -> None:
        r = await self._get_redis()
        if r is not None:
            try:
                await r.set(
                    f"mermaid:memory:{session_id}",
                    memory_text,
                    ex=MEMORY_TTL_SECONDS,
                )
                return
            except Exception as exc:
                logger.warning("Redis set_memory failed: %s", exc)
                self.redis = None
        self._memories[session_id] = memory_text

    async def clear_memory(self, session_id: str) -> None:
        r = await self._get_redis()
        if r is not None:
            try:
                await r.delete(f"mermaid:memory:{session_id}")
            except Exception as exc:
                logger.warning("Redis clear_memory failed: %s", exc)
                self.redis = None
        self._memories.pop(session_id, None)

    def _evict_expired(self, now: float) -> None:
        expired = [
            sid
            for sid, (ts, _) in self._sessions.items()
            if now - ts > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)


class AppState:
    kokoro: Optional[Kokoro] = None
    whisper: Optional[WhisperModel] = None
    http_client: Optional[httpx.AsyncClient] = None
    sessions: SessionStore = SessionStore(MAX_SESSIONS, SESSION_TTL_SECONDS)
    ready: bool = False


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = ensure_file(MODEL_CACHE_DIR / "kokoro-v1.0.onnx", KOKORO_ONNX_URL)
    voices_path = ensure_file(MODEL_CACHE_DIR / "voices-v1.0.bin", KOKORO_VOICES_URL)

    logger.info("Loading Kokoro ONNX TTS model...")
    state.kokoro = Kokoro(str(onnx_path), str(voices_path))

    logger.info("Loading faster-whisper STT model (%s, %s)...", STT_MODEL_NAME, STT_COMPUTE_TYPE)
    whisper_dir = MODEL_CACHE_DIR / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    state.whisper = WhisperModel(
        STT_MODEL_NAME,
        device=os.getenv("STT_DEVICE", "cpu"),
        compute_type=STT_COMPUTE_TYPE,
        download_root=str(whisper_dir),
    )

    await state.sessions._get_redis()
    state.http_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0))
    state.ready = True
    logger.info("%s Voice Agent ready.", AGENT_NAME)
    try:
        yield
    finally:
        state.ready = False
        if state.sessions.redis is not None:
            await state.sessions.redis.aclose()
        if state.http_client is not None:
            await state.http_client.aclose()


app = FastAPI(title="Mermaid Voice Agent", version="1.2.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    if not state.ready or state.kokoro is None or state.whisper is None:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ready"}


@app.get("/", response_class=HTMLResponse)
async def index():
    if not INDEX_HTML_PATH.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))


@app.get("/api/voices")
async def get_voices():
    return {"agent_name": AGENT_NAME, "default": "af_heart", "voices": MERMAID_VOICES}


@app.get("/api/history")
async def get_history(x_session_id: str = Header(default="default")):
    history = await state.sessions.get_history(x_session_id)
    memory = await state.sessions.get_memory(x_session_id)
    return {"history": history, "has_memory": bool(memory)}


@app.post("/api/reset")
async def reset_session(
    forget_memory: bool = Query(default=False),
    x_session_id: str = Header(default="default"),
):
    await state.sessions.clear_history(x_session_id)
    if forget_memory:
        await state.sessions.clear_memory(x_session_id)
    return {"ok": True}


def _transcribe_sync(audio: np.ndarray) -> str:
    assert state.whisper is not None
    if audio.size < 5600:
        return ""
    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 0.004:
        return ""
    peak = float(np.max(np.abs(audio)))
    if 0 < peak < 0.25:
        audio = (audio / peak) * 0.5

    segments, _ = state.whisper.transcribe(
        audio,
        language="en",
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        temperature=0.0,
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    cleaned = re.sub(r"[\[\(].*?[\]\)]", "", text).strip()
    if cleaned.lower() in {"", "you", "thank you.", "thanks for watching.", "bye."}:
        return ""
    return cleaned


def _synthesize_sync(reply_text: str, voice: str, ethereal: bool) -> bytes:
    assert state.kokoro is not None
    lang = "en-gb" if voice.startswith(("bf_", "bm_")) else "en-us"
    samples, sample_rate = state.kokoro.create(
        reply_text, voice=voice, speed=0.94, lang=lang
    )
    audio = np.asarray(samples, dtype=np.float32)
    if ethereal:
        audio = apply_ethereal_polish(audio, sample_rate)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def _ask_llm(history: List[dict], memory: str, user_text: str) -> str:
    assert state.http_client is not None
    system_content = BASE_SYSTEM_PROMPT
    if memory:
        system_content += f"\n\nLong-term memory from past conversations:\n{memory}"

    messages = [{"role": "system", "content": system_content}] + history + [
        {"role": "user", "content": user_text}
    ]
    payload = {
        "model": LLM_MODEL,
        "messages": messages,
        "temperature": 0.6,
        "max_tokens": 120,
    }
    headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
    resp = await state.http_client.post(
        f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers
    )
    resp.raise_for_status()
    data = resp.json()
    reply = data["choices"][0]["message"]["content"].strip()
    return re.sub(r"[*_`#]+", "", reply).strip()


async def _update_long_term_memory(session_id: str, user_text: str, reply_text: str) -> None:
    """Extract durable facts & conversation summary in the background (zero latency)."""
    if state.http_client is None:
        return
    try:
        existing = await state.sessions.get_memory(session_id)
        prompt = (
            "Maintain concise long-term memory notes about the user and past conversations. "
            "Merge any new facts, preferences, names, or topics from the latest exchange into "
            "the existing notes. Keep at most 12 short lines. If nothing new is worth remembering, "
            "return the existing notes unchanged.\n\n"
            f"Existing memory:\n{existing or '(none)'}\n\n"
            f"Latest exchange:\nUser: {user_text}\nAssistant: {reply_text}\n\n"
            "Updated memory notes:"
        )
        payload = {
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 220,
        }
        headers = {"Authorization": f"Bearer {LLM_API_KEY}"}
        resp = await state.http_client.post(
            f"{LLM_BASE_URL}/chat/completions", json=payload, headers=headers
        )
        resp.raise_for_status()
        updated = resp.json()["choices"][0]["message"]["content"].strip()
        if updated:
            await state.sessions.set_memory(session_id, updated)
    except Exception as exc:
        logger.warning("Background memory update failed: %s", exc)


@app.post("/api/talk")
async def talk(
    request: Request,
    voice: str = Query(default="af_heart"),
    ethereal: str = Query(default="1"),
    x_session_id: str = Header(default="default"),
):
    if not state.ready:
        raise HTTPException(status_code=503, detail="Models are still loading")

    if voice not in MERMAID_VOICES:
        voice = "af_heart"
    use_ethereal = ethereal == "1"

    wav_bytes = await request.body()
    if not wav_bytes:
        return {"user_text": "", "reply_text": "", "audio_b64": ""}

    audio, orig_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if orig_sr != 16000:
        g = math.gcd(int(orig_sr), 16000)
        audio = resample_poly(audio, 16000 // g, int(orig_sr) // g).astype(np.float32)

    user_text = await asyncio.to_thread(_transcribe_sync, audio)
    if not user_text:
        return {"user_text": "", "reply_text": "", "audio_b64": ""}

    history = await state.sessions.get_history(x_session_id)
    memory = await state.sessions.get_memory(x_session_id)
    logger.info(
        "session=%s history_len=%d memory_len=%d user=%r",
        x_session_id,
        len(history),
        len(memory),
        user_text,
    )
    try:
        reply_text = await _ask_llm(history, memory, user_text)
    except Exception as exc:
        logger.exception("LLM request failed: %s", exc)
        raise HTTPException(status_code=502, detail="Local LLM service unavailable") from exc

    if not reply_text:
        return {"user_text": user_text, "reply_text": "", "audio_b64": ""}

    await state.sessions.set_history(
        x_session_id,
        history
        + [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply_text},
        ],
    )
    asyncio.create_task(_update_long_term_memory(x_session_id, user_text, reply_text))

    wav_out = await asyncio.to_thread(
        _synthesize_sync, reply_text, voice, use_ethereal
    )
    return {
        "user_text": user_text,
        "reply_text": reply_text,
        "audio_b64": base64.b64encode(wav_out).decode("ascii"),
    }
