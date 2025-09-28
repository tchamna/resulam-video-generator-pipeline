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
import re

import step0_config as cfg


# ── USER SETTINGS ──────────────────────────────────────────────────────

# BASE_DIR        = Path(os.getcwd())
BASE_DIR = cfg.BASE_DIR


LANGUAGE = cfg.LANGUAGE.title()                 # e.g. "Duala"
MODE     = cfg.MODE.lower()                     # "lecture" | "homework"
# Chapter selection (all optional in cfg)
SINGLE_CHAPTER = getattr(cfg, "SINGLE_CHAPTER", None)

START_CHAPTER = getattr(cfg, "START_CHAPTER", 1)
END_CHAPTER   = getattr(cfg, "END_CHAPTER", None)  # None = until last


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
# COMBINED_VIDEO_DIR = Path(r"D:\Resulam\Videos_Production\private_assets\Languages\DualaPhrasebook\Results_Audios\gen3_bilingual_sentences\bilingual_sentences_chapters")

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

        # ---- Detect if it's a video or audio ----
        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))
        if is_video:
            clip = VideoFileClip(str(file_path))
            voice_audio = clip.audio
        else:
            clip = None
            voice_audio = AudioFileClip(str(file_path))

        # ---- Loop and fade background music ----
        loops = int(voice_audio.duration // bg_music.duration) + 1
        bg_looped = concatenate_audioclips([bg_music] * loops).set_duration(voice_audio.duration)
        fade_duration = min(3, voice_audio.duration * 0.1)
        bg_looped = bg_looped.audio_fadein(fade_duration).audio_fadeout(fade_duration)

        # ---- Mix tracks ----
        mixed_audio = CompositeAudioClip([voice_audio, bg_looped])

        # ✅ Make sure fps is set (no need for nchannels)
        audio_fps = getattr(voice_audio, "fps", 44100)
        mixed_audio = mixed_audio.set_fps(audio_fps)

        output_path = Path(output_dir) / f"mixed_{filename}"

        ffmpeg_params = ["-movflags", "faststart"]
        if FFMPEG_THREADS and FFMPEG_THREADS > 0:
            ffmpeg_params += ["-threads", str(FFMPEG_THREADS)]

        if is_video:
            final_clip = clip.set_audio(mixed_audio)
            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                audio_fps=audio_fps,
                remove_temp=True,
                logger="bar",
                ffmpeg_params=ffmpeg_params,
            )
            final_clip.close()
        else:
            mixed_audio.write_audiofile(
                str(output_path),
                fps=audio_fps,
                codec="libmp3lame",      # ✅ correct codec for MP3
                bitrate="192k",
                logger="bar",
                ffmpeg_params=ffmpeg_params,
            )

        mixed_audio.close()
        voice_audio.close()
        bg_music.close()
        if clip:
            clip.close()

        return f"✅ {filename} -> {output_path}"

    except Exception as e:
        return f"❌ {filename} failed: {e}"

def process_file(filename: str, music_path: str, combined_dir: str, output_dir: str):
    try:
        file_path = Path(combined_dir) / filename
        bg_music = AudioFileClip(music_path).volumex(0.1)

        # ---- Detect if it's a video or pure audio ----
        is_video = filename.lower().endswith(('.mp4', '.avi', '.mov'))
        if is_video:
            clip = VideoFileClip(str(file_path))
            voice_audio = clip.audio
        else:
            clip = None
            voice_audio = AudioFileClip(str(file_path))

        # ---- Loop + fade background music ----
        loops = int(voice_audio.duration // bg_music.duration) + 1
        bg_looped = concatenate_audioclips([bg_music] * loops).set_duration(voice_audio.duration)
        fade_duration = min(3, voice_audio.duration * 0.1)
        bg_looped = bg_looped.audio_fadein(fade_duration).audio_fadeout(fade_duration)

        # ---- Mix ----
        mixed_audio = CompositeAudioClip([voice_audio, bg_looped])

        # Ensure FPS (important for audio export)
        audio_fps = getattr(voice_audio, "fps", 44100)
        mixed_audio = mixed_audio.set_fps(audio_fps)

        # ---- Output path ----
        if is_video:
            output_path = Path(output_dir) / f"mixed_{filename}"
        else:
            # always export audio-only as mp3
            output_path = Path(output_dir) / f"mixed_{Path(filename).stem}.mp3"

        # ---- ffmpeg params ----
        ffmpeg_params = ["-movflags", "faststart"]
        if FFMPEG_THREADS and FFMPEG_THREADS > 0:
            ffmpeg_params += ["-threads", str(FFMPEG_THREADS)]

        if is_video:
            final_clip = clip.set_audio(mixed_audio)
            final_clip.write_videofile(
                str(output_path),
                codec="libx264",
                audio_codec="aac",
                audio_fps=audio_fps,
                remove_temp=True,
                logger="bar",
                ffmpeg_params=ffmpeg_params,
            )
            final_clip.close()
        else:
            mixed_audio.write_audiofile(
                str(output_path),
                fps=audio_fps,
                codec="libmp3lame",  # correct codec for MP3
                bitrate="192k",
                logger="bar",
                ffmpeg_params=ffmpeg_params,
            )

        # ---- Close resources ----
        mixed_audio.close()
        voice_audio.close()
        bg_music.close()
        if clip:
            clip.close()

        return f"✅ {filename} -> {output_path}"

    except Exception as e:
        return f"❌ {filename} failed: {e}"


# ── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with log_time("Total processing"):
        
        def get_chapter_num(name: str) -> int | None:
            """
            Try to extract a chapter number from filenames like:
            'duala_chapter_15_chunk_02.mp4'
            Returns int or None if no match.
            """
            m = re.search(r"chapter[_-](\d+)", name, re.IGNORECASE)
            return int(m.group(1)) if m else None

        all_files = [
            f for f in natsorted(os.listdir(COMBINED_VIDEO_DIR))
            if f.lower().endswith(('.mp3', '.mp4', '.avi', '.mov'))
            and "Kiss the Sky" not in f
        ]

        # --- New: filter by chapter start/end if numbers present ---
        filtered = []
        for f in all_files:
            chap = get_chapter_num(f)
            if chap is None:
                # Keep files with no chapter info (safe fallback)
                filtered.append(f)
            else:
                if chap >= START_CHAPTER and (END_CHAPTER is None or chap <= END_CHAPTER):
                    filtered.append(f)

        files = FILES_TO_PROCESS or filtered
        logging.info(f"🎯 Files selected: {len(files)} (chapters {START_CHAPTER}–{END_CHAPTER})")

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
