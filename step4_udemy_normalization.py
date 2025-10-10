from __future__ import annotations
import os, time, re, math, logging, subprocess
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed
from natsort import natsorted
import numpy as np
import pandas as pd
from pydub import AudioSegment

import step0_config as cfg

# ── USER SETTINGS ──────────────────────────────────────────────────────
BASE_DIR        = cfg.BASE_DIR
LANGUAGE        = cfg.LANGUAGE.title()
MODE            = cfg.MODE.lower()
SINGLE_CHAPTER  = getattr(cfg, "SINGLE_CHAPTER", None)

_cfg_start = getattr(cfg, "START_CHAPTER", None)
_cfg_end   = getattr(cfg, "END_CHAPTER", None)
START_CHAPTER = 1 if _cfg_start is None else int(_cfg_start)
END_CHAPTER   = None if _cfg_end is None else int(_cfg_end)

USE_PARALLEL    = False
MAX_WORKERS     = 2
FFMPEG_THREADS  = 2
FILES_TO_PROCESS: list[str] = []

FORCE_REBUILD   = False  # 🔥 Change to True if you want to reprocess existing files

USE_PRIVATE_ASSETS = os.getenv(
    "USE_PRIVATE_ASSETS",
    "1" if getattr(cfg, "USE_PRIVATE_ASSETS", True) else "0"
) == "1"

def get_asset_path(relative: str) -> Path:
    base = BASE_DIR / ("private_assets" if USE_PRIVATE_ASSETS else "assets")
    return base / relative

LANG_BASE_DIR      = get_asset_path(f"Languages/{LANGUAGE}Phrasebook")
VIDEO_DIR          = LANG_BASE_DIR / f"Results_Videos/{MODE.title()}"
COMBINED_VIDEO_DIR = VIDEO_DIR / f"{LANGUAGE}_Chapters_Combined"
OUTPUT_DIR         = COMBINED_VIDEO_DIR / "normalized_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_DIR = LANG_BASE_DIR / "Logs"
LOG_DIR.mkdir(exist_ok=True)

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
def format_elapsed(sec: float) -> str:
    return f"{sec:.2f}s" if sec < 60 else f"{sec/60:.2f}min"

from contextlib import contextmanager
@contextmanager
def log_time(label: str):
    start = time.perf_counter()
    logging.info(f"▶️ {label} …")
    yield
    elapsed = time.perf_counter() - start
    logging.info(f"⏱ Finished {label} in {format_elapsed(elapsed)}")

# ── AUDIO HELPERS ──────────────────────────────────────────────────────
def get_audio_stats(video_path: Path) -> dict:
    audio = AudioSegment.from_file(video_path)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= float(2 ** (8 * audio.sample_width - 1))
    rms = np.sqrt(np.mean(samples**2))
    peak = np.max(np.abs(samples))
    dbfs = 20 * math.log10(rms + 1e-10)
    return {
        "duration_sec": len(audio)/1000,
        "channels": audio.channels,
        "frame_rate": audio.frame_rate,
        "rms": float(rms),
        "peak": float(peak),
        "rms_dbfs": float(dbfs)
    }

def measure_loudness_dbfs(audio: AudioSegment) -> float:
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= float(2 ** (8 * audio.sample_width - 1))
    rms = np.sqrt(np.mean(samples**2))
    return 20 * math.log10(rms + 1e-10)

def normalize_to_target(audio: AudioSegment, target_dbfs=-16.0) -> AudioSegment:
    change = target_dbfs - measure_loudness_dbfs(audio)
    return audio.apply_gain(change)

def normalize_video_audio(
    video_path: Path,
    output_dir: Path,
    target_dbfs=-16.0,
    suffix="_normalized",
    force: bool = False
) -> Path:
    out = output_dir / f"{video_path.stem}{suffix}{video_path.suffix}"
    if out.exists() and not force:
        logging.info(f"⚡ Skipping {out.name} (already normalized)")
        return out

    temp = video_path.with_suffix(".temp.wav")
    audio = AudioSegment.from_file(video_path)
    audio_norm = normalize_to_target(audio, target_dbfs)
    audio_norm.export(temp, format="wav")

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(temp),
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(out)
    ]
    if FFMPEG_THREADS:
        cmd.insert(-1, "-threads")
        cmd.insert(-1, str(FFMPEG_THREADS))

    logging.info(f"🎧 Normalizing {video_path.name} → {out.name}")
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    temp.unlink(missing_ok=True)
    return out

# ── FILE SELECTION ─────────────────────────────────────────────────────
def get_chapter_num(name: str) -> int | None:
    m = re.search(r"(?:chapter|chap)[_-]?\s*(\d+)", name, re.IGNORECASE)
    return int(m.group(1)) if m else None

def select_files() -> list[Path]:
    all_files = [
        Path(COMBINED_VIDEO_DIR) / f
        for f in natsorted(os.listdir(COMBINED_VIDEO_DIR))
        if f.lower().endswith(('.mp4', '.mov', '.avi'))
    ]
    chosen = []
    for f in all_files:
        chap = get_chapter_num(f.name)
        if chap is None or (chap >= START_CHAPTER and (END_CHAPTER is None or chap <= END_CHAPTER)):
            chosen.append(f)
    return [Path(x) for x in FILES_TO_PROCESS] if FILES_TO_PROCESS else chosen

# ── MAIN ───────────────────────────────────────────────────────────────

def main():
    files = select_files()
    logging.info(f"🎯 {len(files)} files selected")

    csv = OUTPUT_DIR / "audio_comparison.csv"
    # Remove old CSV if you want a fresh one each run
    if csv.exists():
        csv.unlink()

    header_written = False

    def work(f: Path):
        logging.info(f"🎬 Processing {f.name}")
        before = get_audio_stats(f)
        out = normalize_video_audio(f, OUTPUT_DIR, force=FORCE_REBUILD)
        after = get_audio_stats(out)
        return [
            {"file": f.name, "stage": "Before", **before},
            {"file": out.name, "stage": "After", **after},
        ]

    with log_time("Total normalization"):
        if USE_PARALLEL:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
                for res in ex.map(work, files):
                    df_chunk = pd.DataFrame(res)
                    df_chunk.to_csv(csv, mode="a", header=not header_written, index=False)
                    header_written = True
        else:
            for f in files:
                res = work(f)
                df_chunk = pd.DataFrame(res)
                df_chunk.to_csv(csv, mode="a", header=not header_written, index=False)
                header_written = True

    logging.info(f"✅ Summary saved to {csv}")
    # df = pd.read_csv(csv)

if __name__ == "__main__":
    main()
