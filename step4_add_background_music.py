from __future__ import annotations
import os, time
from pathlib import Path
from natsort import natsorted
from moviepy.editor import (
    VideoFileClip,
    AudioFileClip,
    concatenate_audioclips,
    CompositeAudioClip
)
import logging
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import step0_config as cfg


# ── USER SETTINGS ──────────────────────────────────────────────────────

# BASE_DIR        = Path(os.getcwd())
BASE_DIR = cfg.BASE_DIR


LANGUAGE = cfg.LANGUAGE.title()                 # e.g. "Duala"
MODE     = cfg.MODE.lower()                     # "lecture" | "homework"


USE_PARALLEL    = True    # 🔁 Set to False for sequential processing
USE_PARALLEL    = False    # 🔁 Set to False for sequential processing
MAX_WORKERS     = 2       # Python jobs in parallel
FFMPEG_THREADS  = 2       # Threads per ffmpeg process (set 0 or None for auto)

FILES_TO_PROCESS = []     # Leave empty to process all


# if "USE_PRIVATE_ASSETS" in os.environ:
#     USE_PRIVATE_ASSETS = os.getenv("USE_PRIVATE_ASSETS") == "1"
#     print("using Private Assets from env variable")
# else:
#     print(" Private Assets not found from the env variable")
#     USE_PRIVATE_ASSETS = True

# Env override wins, otherwise fallback to cfg.USE_PRIVATE_ASSETS
USE_PRIVATE_ASSETS = os.getenv(
    "USE_PRIVATE_ASSETS",
    "1" if getattr(cfg, "USE_PRIVATE_ASSETS", True) else "0"
) == "1"

def get_asset_path(relative_path: str) -> Path:
    base = BASE_DIR / ("private_assets" if USE_PRIVATE_ASSETS else "assets")
    return base / relative_path


# ── FOLDER LAYOUT ──────────────────────────────────────────────────────
LANG_BASE_DIR = get_asset_path(f"Languages/{LANGUAGE.title()}Phrasebook")
VIDEO_DIR     = get_asset_path(f"{LANG_BASE_DIR}/Results_Videos/{MODE.title()}")
COMBINED_VIDEO_DIR = VIDEO_DIR / f"{LANGUAGE.title()}_Chapters_Combined"
MUSIC_PATH    = LANG_BASE_DIR / "duala_music_background.mp3"

output_folder = COMBINED_VIDEO_DIR / "mixed_output"
os.makedirs(output_folder, exist_ok=True)

LOG_DIR = LANG_BASE_DIR / "Logs"
os.makedirs(LOG_DIR, exist_ok=True)


# ── LOGGING ────────────────────────────────────────────────────────────
log_file = LOG_DIR / Path(__file__).with_suffix(".log").name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# ── TIMER HELPERS ──────────────────────────────────────────────────────
def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} sec"
    elif seconds < 3600:
        return f"{seconds/60:.2f} min"
    elif seconds < 86400:
        return f"{seconds/3600:.2f} hr"
    else:
        return f"{seconds/86400:.2f} days"


@contextmanager
def log_time(step_name: str):
    start = time.perf_counter()
    logging.info(f"▶️ Starting {step_name}…")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"⏱ Finished {step_name} in {format_elapsed(elapsed)}")


# ── WORKER FUNCTION ────────────────────────────────────────────────────
def process_file(filename: str, music_path: str, combined_dir: str, output_dir: str):
    try:
        file_path = Path(combined_dir) / filename
        bg_music = AudioFileClip(music_path).volumex(0.1)

        # Load video/audio
        if filename.endswith(('.mp4', '.avi', '.mov')):
            clip = VideoFileClip(str(file_path))
            voice_audio = clip.audio
        else:
            clip = None
            voice_audio = AudioFileClip(str(file_path))

        # Prepare background music
        loops = int(voice_audio.duration // bg_music.duration) + 1
        bg_looped = concatenate_audioclips([bg_music] * loops).set_duration(voice_audio.duration)
        fade_duration = min(3, voice_audio.duration * 0.1)
        bg_looped = bg_looped.audio_fadein(fade_duration).audio_fadeout(fade_duration)

        # Mix
        mixed_audio = CompositeAudioClip([voice_audio, bg_looped])

        # Export
        output_path = Path(output_dir) / f"mixed_{filename}"

        # ffmpeg params
        ffmpeg_params = ["-movflags", "faststart"]
        if FFMPEG_THREADS and FFMPEG_THREADS > 0:
            ffmpeg_params += ["-threads", str(FFMPEG_THREADS)]

        if clip:
            final_clip = clip.set_audio(mixed_audio)
            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                remove_temp=True,   # ✅ auto-clean temp files
                logger="bar",       # ✅ show progress bar
                ffmpeg_params=ffmpeg_params
            )
            final_clip.close()
        else:
            mixed_audio.write_audiofile(
                str(output_path),
                codec="mp3",
                logger="bar"
            )

        # Close resources
        voice_audio.close()
        bg_music.close()

        return f"✅ {filename} -> {output_path}"

    except Exception as e:
        return f"❌ {filename} failed: {e}"


# ── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with log_time("Total processing"):
        all_files = [
            f for f in natsorted(os.listdir(COMBINED_VIDEO_DIR))
            if f.endswith(('.mp3', '.mp4', '.avi', '.mov')) and "Kiss the Sky" not in f
        ]

        # If user provided a specific list, filter down
        if FILES_TO_PROCESS:
            files = [f for f in all_files if f in FILES_TO_PROCESS]
        else:
            files = all_files

        logging.info(f"🎯 Files selected: {len(files)}")

        if USE_PARALLEL:
            workers = max(1, MAX_WORKERS)
            logging.info(f"⚡ Parallel mode ON → using {workers} workers (threads)")
            results = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_file = {
                    executor.submit(process_file, f, str(MUSIC_PATH), str(COMBINED_VIDEO_DIR), str(output_folder)): f
                    for f in files
                }
                for future in as_completed(future_to_file):
                    result = future.result()
                    logging.info(result)
                    results.append(result)
        else:
            logging.info("🐢 Sequential mode ON")
            results = []
            for f in files:
                result = process_file(f, str(MUSIC_PATH), str(COMBINED_VIDEO_DIR), str(output_folder))
                logging.info(result)
                results.append(result)

        logging.info("🏁 All files processed")

        # ── CLEANUP TEMP FILES ───────────────────────────────────────────────
        for ext in ("*.mp3", "*.m4a"):
            for f in BASE_DIR.glob(ext):
                if "MPY_wvf_snd" in f.name or f.name.startswith("temp_audio_"):
                    try:
                        f.unlink()
                        logging.info(f"🗑️ Deleted temp file {f}")
                    except Exception as e:
                        logging.warning(f"Could not delete {f}: {e}")
