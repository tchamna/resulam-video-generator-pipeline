from __future__ import annotations
import os, time, re, logging, subprocess
from pathlib import Path
from natsort import natsorted
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment

import step0_config as cfg

# ── USER SETTINGS ──────────────────────────────────────────────────────
BASE_DIR        = cfg.BASE_DIR
LANGUAGE        = cfg.LANGUAGE.title()
MODE            = cfg.MODE.lower()
_cfg_start      = getattr(cfg, "START_CHAPTER", None)
_cfg_end        = getattr(cfg, "END_CHAPTER", None)
START_CHAPTER   = 1 if _cfg_start is None else int(_cfg_start)
END_CHAPTER     = None if _cfg_end is None else int(_cfg_end)

USE_PARALLEL    = False
MAX_WORKERS     = 2
FFMPEG_THREADS  = 2
FILES_TO_PROCESS: list[str] = []     # Leave empty to process all

USE_PRIVATE_ASSETS = os.getenv(
    "USE_PRIVATE_ASSETS",
    "1" if getattr(cfg, "USE_PRIVATE_ASSETS", True) else "0"
) == "1"

def get_asset_path(relative: str) -> Path:
    base = BASE_DIR / ("private_assets" if USE_PRIVATE_ASSETS else "assets")
    return base / relative

LANG_BASE_DIR       = get_asset_path(f"Languages/{LANGUAGE}Phrasebook")
VIDEO_DIR           = LANG_BASE_DIR / f"Results_Videos/{MODE.title()}"
COMBINED_VIDEO_DIR  = VIDEO_DIR / f"{LANGUAGE}_Chapters_Combined"
NORMALIZED_DIR      = COMBINED_VIDEO_DIR / "normalized_output"
OUTPUT_DIR          = COMBINED_VIDEO_DIR / "normalized_with_bg"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MUSIC_FILENAME = getattr(cfg, "MUSIC_FILENAME", f"{LANGUAGE.lower()}_music_background.mp3")
MUSIC_GAIN_DB  = float(getattr(cfg, "MUSIC_GAIN_DB", -25.0))
MUSIC_PATH     = LANG_BASE_DIR / MUSIC_FILENAME
if not MUSIC_PATH.exists():
    raise FileNotFoundError(f"Background music file not found: {MUSIC_PATH}")

LOG_DIR = LANG_BASE_DIR / "Logs"
LOG_DIR.mkdir(exist_ok=True)

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

# ── HELPERS ────────────────────────────────────────────────────────────
def format_elapsed(sec: float) -> str:
    return f"{sec:.2f}s" if sec < 60 else f"{sec/60:.2f}min"

def get_chapter_num(name: str) -> int | None:
    m = re.search(r"(?:chapter|chap)[_-]?\s*(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else None

def select_files() -> list[Path]:
    all_files = [
        Path(NORMALIZED_DIR) / f
        for f in natsorted(os.listdir(NORMALIZED_DIR))
        if f.lower().endswith(('.mp3', '.wav', '.mp4', '.mov', '.avi'))
    ]
    chosen = []
    for f in all_files:
        chap = get_chapter_num(f.name)
        if chap is None or (chap >= START_CHAPTER and (END_CHAPTER is None or chap <= END_CHAPTER)):
            chosen.append(f)
    return [Path(x) for x in FILES_TO_PROCESS] if FILES_TO_PROCESS else chosen

def add_bg_music_to_video(normalized_video: Path, music_file: Path, out_dir: Path, music_gain_db=-25.0) -> Path:
    """Overlay music under normalized video and keep video stream."""
    speech = AudioSegment.from_file(normalized_video)
    music  = AudioSegment.from_file(music_file) + music_gain_db
    loops = int(speech.duration_seconds // music.duration_seconds) + 1
    music_looped = (music * loops)[: len(speech)]
    mixed = speech.overlay(music_looped)

    temp_mixed = normalized_video.with_suffix(".temp.wav")
    mixed.export(temp_mixed, format="wav")

    out_file = out_dir / f"{normalized_video.stem}_bg{normalized_video.suffix}"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(normalized_video),
        "-i", str(temp_mixed),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(out_file)
    ]
    if FFMPEG_THREADS:
        cmd.insert(-1, "-threads")
        cmd.insert(-1, str(FFMPEG_THREADS))
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    temp_mixed.unlink(missing_ok=True)
    return out_file

def add_bg_music_to_audio(normalized_audio: Path, music_file: Path, out_dir: Path, music_gain_db=-25.0) -> Path:
    """Overlay music under normalized audio (MP3/WAV)."""
    speech = AudioSegment.from_file(normalized_audio)
    music  = AudioSegment.from_file(music_file) + music_gain_db
    loops = int(speech.duration_seconds // music.duration_seconds) + 1
    music_looped = (music * loops)[: len(speech)]
    mixed = speech.overlay(music_looped)

    out_file = out_dir / f"{normalized_audio.stem}_bg{normalized_audio.suffix}"
    mixed.export(out_file, format=normalized_audio.suffix.lstrip('.'))
    return out_file

# ── MAIN ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    start_time = time.perf_counter()
    files = select_files()
    logging.info(f"🎯 {len(files)} normalized files selected (chapters {START_CHAPTER}–{END_CHAPTER})")

    def process(f: Path):
        try:
            logging.info(f"🎬 Adding background music: {f.name}")
            if f.suffix.lower() in ['.mp4', '.mov', '.avi']:
                out = add_bg_music_to_video(f, MUSIC_PATH, OUTPUT_DIR, music_gain_db=MUSIC_GAIN_DB)
            else:
                out = add_bg_music_to_audio(f, MUSIC_PATH, OUTPUT_DIR, music_gain_db=MUSIC_GAIN_DB)
            return f"✅ {f.name} -> {out.name}"
        except Exception as e:
            return f"❌ {f.name} failed: {e}"

    if USE_PARALLEL:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            for result in ex.map(process, files):
                logging.info(result)
    else:
        for f in files:
            logging.info(process(f))

    logging.info(f"🏁 Completed in {format_elapsed(time.perf_counter()-start_time)}")
