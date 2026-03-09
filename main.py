from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import shutil, os, uuid, logging, time
import ffmpeg
from diarization_pipeline import process_audio_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

app = FastAPI(title="Speaker Diarization API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".mp4"}   # ✅ wider support

@app.post("/process-audio")
async def process_audio(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    _, ext = os.path.splitext(file.filename)
    if ext.lower() not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTS}")

    file_id  = str(uuid.uuid4())
    raw_path = os.path.join(UPLOAD_DIR, f"{file_id}{ext}")
    wav_path = os.path.join(UPLOAD_DIR, f"{file_id}.wav")

    try:
        logging.info(f"📥 Received: {file.filename}")
        t0 = time.time()

        content = await file.read()
        with open(raw_path, "wb") as f:
            f.write(content)
        logging.info(f"💾 Saved {len(content)/1024:.1f} KB — converting to WAV...")

        # ✅ Always re-encode to mono 16kHz PCM — both models need this
        try:
            (
                ffmpeg
                .input(raw_path)
                .output(wav_path, acodec="pcm_s16le", ac=1, ar="16000")
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise HTTPException(500, f"FFmpeg error: {e.stderr.decode()}")

        logging.info("✅ WAV ready — running models...")
        transcript = await process_audio_file(wav_path)

        elapsed = time.time() - t0
        logging.info(f"🎉 Complete in {elapsed:.1f}s — {len(transcript)} segments")

        return {
            "status":    "ok",
            "elapsed_s": round(elapsed, 2),
            "segments":  len(transcript),
            "transcript": transcript      # ✅ consistent key name
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Unhandled error")
        raise HTTPException(500, str(e))
    finally:
        background_tasks.add_task(_cleanup, [raw_path, wav_path])


def _cleanup(paths):
    for p in paths:
        try:
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass