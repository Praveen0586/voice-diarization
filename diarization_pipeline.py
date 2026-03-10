import os
import asyncio
import logging
import time
import subprocess
import json
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import login
import torch

# ✅ Authenticate BEFORE importing pyannote — avoids all token-passing issues
_hf_token = os.environ.get("HF_TOKEN", "")
if _hf_token:
    login(token=_hf_token, add_to_git_credential=False)

from pyannote.audio import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

if not _hf_token:
    logging.warning("HF_TOKEN not set. Pyannote may fail.")

# ── Config ─────────────────────────────────────────────────────────────────────
WHISPER_CLI       = "/usr/local/bin/whisper-cli"
WHISPER_MODEL     = os.path.expanduser("~/.cache/whisper/ggml-medium.bin")
INFERENCE_TIMEOUT = int(os.environ.get("INFERENCE_TIMEOUT", "600"))

# ── Load Pyannote ──────────────────────────────────────────────────────────────
try:
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    logging.info(f"Loading Pyannote on: {device}")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1"
    )
    if device != "cpu":
        diarization_pipeline.to(torch.device(device))
    logging.info("✅ Pyannote loaded.")
except Exception as e:
    logging.error(f"Pyannote load failed: {e}")
    diarization_pipeline = None

# ── Verify whisper-cli ─────────────────────────────────────────────────────────
if not os.path.exists(WHISPER_CLI):
    logging.error(f"whisper-cli not found at {WHISPER_CLI}")
    whisper_ok = False
else:
    logging.info(f"✅ whisper-cli found at {WHISPER_CLI}")
    whisper_ok = True

# ── Dedicated thread pools — one each, no contention ──────────────────────────
_whisper_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
_diarize_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="diarize")


async def process_audio_file(file_path: str):
    if diarization_pipeline is None or not whisper_ok:
        raise RuntimeError("Models not fully initialized.")

    loop = asyncio.get_event_loop()
    t_start = time.time()

    logging.info("🎙️  Launching Whisper + Pyannote in parallel...")

    whisper_task     = loop.run_in_executor(_whisper_executor, _run_whisper, file_path)
    diarization_task = loop.run_in_executor(_diarize_executor, _run_diarization, file_path)

    try:
        words, diarization = await asyncio.wait_for(
            asyncio.gather(whisper_task, diarization_task),
            timeout=INFERENCE_TIMEOUT
        )
    except asyncio.TimeoutError:
        raise RuntimeError(f"Timed out after {INFERENCE_TIMEOUT}s.")

    logging.info(f"✅ Both done in {time.time() - t_start:.1f}s — merging...")

    if not words:
        logging.warning("⚠️  Whisper returned 0 words.")

    merged = _merge_results(words, diarization)
    logging.info(f"🏁 Complete — {len(merged)} segments in {time.time() - t_start:.1f}s")
    return merged


def _run_whisper(file_path: str) -> list:
    logging.info("📝 [Whisper] Starting...")
    t = time.time()

    base_out = file_path.replace(".wav", "")
    json_out = base_out + ".json"

    cmd = [
        WHISPER_CLI,
        "--model",       WHISPER_MODEL,
        "--file",        file_path,
        "--output-json",
        "--output-file", base_out,
        "--word-thold",  "0.01",
        "--max-len",     "1",
        "--threads",     "16",
        "--language",    "auto",
        "--beam-size",   "1",
        "--best-of",     "1",
        "--no-fallback",
    ]

    subprocess.run(cmd, capture_output=True, text=True)

    if not os.path.exists(json_out):
        logging.error(f"❌ JSON not found: {json_out}")
        return []

    with open(json_out, "r") as f:
        data = json.load(f)

    try:
        os.remove(json_out)
    except Exception:
        pass

    words = []
    for segment in data.get("transcription", []):
        tokens = segment.get("tokens", [])
        if tokens:
            for token in tokens:
                text = token.get("text", "").strip()
                if not text or text.startswith("["):
                    continue
                offsets = token.get("offsets", {})
                words.append({
                    "start": round(offsets.get("from", 0) / 1000, 3),
                    "end":   round(offsets.get("to", 0) / 1000, 3),
                    "text":  text
                })
        else:
            text = segment.get("text", "").strip()
            if text:
                offsets = segment.get("offsets", {})
                words.append({
                    "start": round(offsets.get("from", 0) / 1000, 3),
                    "end":   round(offsets.get("to", 0) / 1000, 3),
                    "text":  text
                })

    logging.info(f"✅ [Whisper] Done in {time.time()-t:.1f}s — {len(words)} words")
    return words


def _run_diarization(file_path: str) -> list:
    logging.info("🔊 [Pyannote] Starting...")
    t = time.time()
    result = diarization_pipeline(file_path)

    speaker_segments = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        speaker_segments.append({
            "start":   round(turn.start, 3),
            "end":     round(turn.end, 3),
            "speaker": speaker
        })

    logging.info(f"✅ [Pyannote] Done in {time.time()-t:.1f}s — {len(speaker_segments)} turns")
    return speaker_segments


def _merge_results(words: list, speaker_segments: list) -> list:
    logging.info("🔀 [Merge] Aligning words with speaker labels...")

    if not words:
        return []

    aligned_words = []
    for word in words:
        ws, we = word["start"], word["end"]
        best_speaker, max_overlap = "Unknown", 0.0

        for seg in speaker_segments:
            overlap = max(0.0, min(we, seg["end"]) - max(ws, seg["start"]))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = seg["speaker"]

        aligned_words.append({
            "start":   ws,
            "end":     we,
            "text":    word["text"],
            "speaker": best_speaker
        })

    grouped     = []
    cur_speaker = aligned_words[0]["speaker"]
    cur_words   = [aligned_words[0]["text"]]
    cur_start   = aligned_words[0]["start"]
    cur_end     = aligned_words[0]["end"]

    for w in aligned_words[1:]:
        if w["speaker"] == cur_speaker:
            cur_words.append(w["text"])
            cur_end = w["end"]
        else:
            grouped.append({
                "speaker": cur_speaker,
                "start":   cur_start,
                "end":     cur_end,
                "text":    " ".join(cur_words)
            })
            cur_speaker = w["speaker"]
            cur_words   = [w["text"]]
            cur_start   = w["start"]
            cur_end     = w["end"]

    grouped.append({
        "speaker": cur_speaker,
        "start":   cur_start,
        "end":     cur_end,
        "text":    " ".join(cur_words)
    })

    return grouped