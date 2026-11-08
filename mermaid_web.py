#!/usr/bin/env python3
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse, base64, io, json, math, subprocess, tempfile, threading, webbrowser
from pathlib import Path
from urllib.parse import parse_qs
from wsgiref.simple_server import make_server
import numpy as np, soundfile as sf
from scipy.signal import resample_poly
from mermaid_agent import DEFAULT_LLM_MODEL, DEFAULT_STT_MODEL, generate_reply, load_llm, load_stt, transcribe
from mermaid_tts import DEFAULT_MODEL_ID, LOCAL_MODEL_DIR, MERMAID_VOICES, load_kokoro, synthesize

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    tts_model = load_kokoro(str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else DEFAULT_MODEL_ID)
    stt_model = load_stt(DEFAULT_STT_MODEL)
    tokenizer, llm_model, device = load_llm(DEFAULT_LLM_MODEL)
    history, lock = [], threading.Lock()

    def app(environ, start_response):
        nonlocal history
        path, method = environ.get("PATH_INFO", "/"), environ.get("REQUEST_METHOD", "GET")
        def reply_json(payload: dict):
            body = json.dumps(payload).encode("utf-8")
            start_response("200 OK", [("Content-Type", "application/json; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]
        if method == "GET" and path == "/":
            body = INDEX_HTML.read_bytes()
            start_response("200 OK", [("Content-Type", "text/html; charset=utf-8"), ("Content-Length", str(len(body)))])
            return [body]
        if method == "GET" and path == "/api/voices":
            return reply_json({"default": "af_heart", "voices": MERMAID_VOICES})
        if method == "POST" and path == "/api/reset":
            with lock: history = []
            return reply_json({"ok": True})
        if method == "POST" and path == "/api/talk":
            qs = parse_qs(environ.get("QUERY_STRING", ""))
            voice = qs.get("voice", ["af_heart"])[0]
            if voice not in MERMAID_VOICES: voice = "af_heart"
            ethereal = qs.get("ethereal", ["1"])[0] == "1"
            wav_bytes = environ["wsgi.input"].read(int(environ.get("CONTENT_LENGTH") or 0))
            audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
            if audio.ndim > 1: audio = np.mean(audio, axis=1)
            if sr != 16000:
                g = math.gcd(int(sr), 16000)
                audio = resample_poly(audio, 16000 // g, int(sr) // g).astype(np.float32)
            with lock:
                user_text = transcribe(stt_model, audio)
                if not user_text: return reply_json({"user_text": "", "reply_text": "", "audio_b64": ""})
                reply_text, history = generate_reply(tokenizer, llm_model, device, history, user_text)
                if not reply_text: return reply_json({"user_text": user_text, "reply_text": "", "audio_b64": ""})
                wav_out, out_sr = synthesize(tts_model, reply_text, voice=voice, speed=0.94, ethereal=ethereal)
                if wav_out.size > 0:
                    peak = float(np.max(np.abs(wav_out)))
                    print(f"Speaking ({len(wav_out)/out_sr:.1f}s, peak={peak:.2f})...", flush=True)
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp_path = tmp.name
                    try:
                        sf.write(tmp_path, wav_out, out_sr, format="WAV", subtype="PCM_16")
                        subprocess.run(["afplay", tmp_path], check=False)
                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass
                return reply_json({"user_text": user_text, "reply_text": reply_text, "played_server": True})
        start_response("404 Not Found", [("Content-Type", "text/plain")])
        return [b"Not Found"]

    url = f"http://127.0.0.1:{args.port}"
    print(f"Mermaid Voice Agent UI running at: {url}")
    threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    with make_server("127.0.0.1", args.port, app) as httpd:
        httpd.serve_forever()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
