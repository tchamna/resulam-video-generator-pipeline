
from __future__ import annotations
import os
import sys
import time
import logging
import subprocess
from contextlib import contextmanager

import shutil
import threading
from pathlib import Path
from uuid import uuid4
from typing import List, Dict

from PIL import Image, ImageFile, ImageDraw, ImageFont, ImageColor
import numpy as np
from moviepy.editor import (
    AudioFileClip, CompositeAudioClip, CompositeVideoClip,
    ImageClip, TextClip,
)

import step0_config as cfg

from moviepy.config import change_settings


def _configure_stdio_utf8() -> None:
    # Avoid UnicodeEncodeError on Windows consoles (can crash mid-ffmpeg and corrupt MP4s).
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is None:
            continue
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()

# Text rendering backend:
# - Default: disable ImageMagick so we consistently use the bundled TTF fonts (Charis SIL)
#   via the Pillow renderer below.
# - To force ImageMagick/TextClip, set USE_IMAGEMAGICK=1 (optionally also IMAGEMAGICK_BINARY).
USE_IMAGEMAGICK = os.getenv("USE_IMAGEMAGICK", "0") == "1"
if USE_IMAGEMAGICK:
    _im_bin = os.getenv("IMAGEMAGICK_BINARY")
    if _im_bin:
        change_settings({"IMAGEMAGICK_BINARY": _im_bin})
    elif shutil.which("magick"):
        change_settings({"IMAGEMAGICK_BINARY": "magick"})
else:
    change_settings({"IMAGEMAGICK_BINARY": None})

_WARNED_FORCE_TEXTCLIP_FALLBACK = False

# # ─── Asset Source Config ──────────────────────────────
# USE_PRIVATE_ASSETS = True   # switch here: True → private_assets, False → normal assets
# USE_PRIVATE_ASSETS = False   # switch here: True → private_assets, False → normal assets

# BASE_DIR = Path(os.getcwd())

# # Check if env variable exists, otherwise set default
# if "USE_PRIVATE_ASSETS" in os.environ:
#     USE_PRIVATE_ASSETS = os.getenv("USE_PRIVATE_ASSETS") == "1"
#     print("using Private Assets from env variable")
# else:
#     print(" Private Assets not found from the env variable")
#     # Local default when not provided by runner
#     USE_PRIVATE_ASSETS = True   # change default if needed

BASE_DIR = cfg.BASE_DIR
USE_PRIVATE_ASSETS = os.getenv("USE_PRIVATE_ASSETS", "1" 
                               if cfg.USE_PRIVATE_ASSETS 
                               else "0") == "1"

# ── USER SETTINGS ───────────────────────────────────────────────────────
# LANGUAGE = "Duala"          # e.g., "Nufi", "Yoruba", "Duala"
# MODE = "lecture"            # "lecture" or "homework"

# MODE = "homework"            # "lecture" or "homework"

# REBUILD_ALL = False         # Force regeneration of existing videos
# REBUILD_ALL = True         # Force regeneration of existing videos

LANGUAGE = cfg.LANGUAGE.title()
MODE = cfg.MODE.lower()              # "lecture" | "homework"
REBUILD_ALL = bool(getattr(cfg, "REBUILD_ALL", False))
_env_rebuild = os.getenv("REBUILD_ALL", os.getenv("FORCE_REBUILD"))
if _env_rebuild is not None:
    REBUILD_ALL = str(_env_rebuild).strip().lower() not in ("0", "false", "no", "off", "")

# Audio sample rate for MP4 outputs (Windows Media Player tends to be happiest with 48kHz).
AUDIO_SAMPLE_RATE = int(getattr(cfg, "AUDIO_SAMPLE_RATE", 48000))

# Optional post-process remux to improve compatibility with Windows Media Player.
POSTPROCESS_MP4 = bool(getattr(cfg, "POSTPROCESS_MP4", True))
_env_post = os.getenv("POSTPROCESS_MP4")
if _env_post is not None:
    POSTPROCESS_MP4 = str(_env_post).strip().lower() not in ("0", "false", "no", "off", "")

# ── PARALLELISM CONFIG ──────────────────────────────────────────────────
# FFMPEG_THREADS = 4
# MAX_PARALLEL_JOBS = 5

FFMPEG_THREADS = int(getattr(cfg, "FFMPEG_THREADS", 4))
# Backwards-compatible config: use MAX_PARALLEL_JOBS if provided, else fall back to cfg.MAX_WORKERS
MAX_PARALLEL_JOBS = int(getattr(cfg, "MAX_PARALLEL_JOBS", getattr(cfg, "MAX_WORKERS", 4)))

AVAILABLE_CPUS = os.cpu_count() or 1

while FFMPEG_THREADS > AVAILABLE_CPUS:
    FFMPEG_THREADS = max(1, FFMPEG_THREADS // 2)
MAX_PARALLEL_JOBS = min(MAX_PARALLEL_JOBS, AVAILABLE_CPUS // FFMPEG_THREADS)



def get_asset_path(relative_path: str) -> Path:
    """
    Resolve a path inside either 'assets' or 'private_assets'
    depending on USE_PRIVATE_ASSETS flag.
    """
    base = BASE_DIR / ("private_assets" if USE_PRIVATE_ASSETS else "assets")
    return base / relative_path

# ── PATHS SETUP ─────────────────────────────────────────────────────
# FONT_PATH = BASE_DIR / "assets"/ "Fonts" / "arialbd.ttf"
# LOGO_PATH = BASE_DIR/ "assets" / "resulam_logo_resurrectionLangue.png"

def _resolve_font_path(cfg_value, default_relative: str) -> Path:
    # If config provided an absolute/existing path, prefer it; otherwise treat as relative.
    if cfg_value:
        candidate = Path(cfg_value)
        if candidate.exists():
            return candidate
        return get_asset_path(str(cfg_value))
    return get_asset_path(default_relative)

cfg_font_basic = getattr(cfg, "FONT_PATH_BASIC", None) or getattr(cfg, "FONT_PATH", None)
cfg_font_special = getattr(cfg, "FONT_PATH_SPECIAL", None) or getattr(cfg, "FONT_PATH", None)

BASIC_FONT_PATH = _resolve_font_path(cfg_font_basic, "Fonts/CharisSIL-B.ttf")
SPECIAL_FONT_PATH = _resolve_font_path(cfg_font_special, "Fonts/arialbd.ttf")

# Backward-compatible default (tools may import FONT_PATH directly)
FONT_PATH = BASIC_FONT_PATH

LOGO_PATH = get_asset_path("resulam_logo_resurrectionLangue.png")

# assets_dir = BASE_DIR / "assets"
# LOG_DIR = assets_dir / "Languages" / f"{LANGUAGE.title()}Phrasebook"/"Logs"

LOG_DIR = get_asset_path(f"Languages/{LANGUAGE.title()}Phrasebook/Logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

# MODE_FOLDER = "Lecture" if MODE.lower() == "lecture" else "Homework"



VIDEO_OUT_DIR = get_asset_path(f"Languages/{LANGUAGE.title()}Phrasebook/Results_Videos/{MODE.title()}")
VIDEO_OUT_DIR.mkdir(parents=True, exist_ok=True)



BACKGROUND_DIR = get_asset_path(f"Backgrounds_Selected/{LANGUAGE.title()}")
LOCAL_AUDIO_DIR = get_asset_path(f"Languages/{LANGUAGE.title()}Phrasebook/Results_Audios/{MODE}_gen2_normalized_padded")
ENG_AUDIO_DIR = get_asset_path("EnglishOnly")
 # english_audio_path = BASE_DIR/"assets"/"EnglishOnly" / sentence["english_audio"]
   
SENTENCES_PATH = get_asset_path(f"Languages/{LANGUAGE.title()}Phrasebook/{LANGUAGE.lower()}_english_french_phrasebook_sentences_list.txt")

# ── FONTS SETUP ─────────────────────────────────────────────────────

# VIDEO_RESOLUTION = (1920, 1080)
# FRAME_RATE = 24
# INNER_PAUSE_DURATION = 5
# TRAILING_PAUSE_DURATION = 3
# DEFAULT_FONT_SIZE = 100
# HEADER_GAP_RATIO = 0.30
# MARGIN_RIGHT = 25
# MARGIN_TOP = 15

# PROJECT_MODE = "Test"  # or "Production"
# PROJECT_MODE = "Production"  # or "Production"

VIDEO_RESOLUTION = tuple(getattr(cfg, "VIDEO_RESOLUTION", (1920, 1080)))
FRAME_RATE = int(getattr(cfg, "FRAME_RATE", 24))
INNER_PAUSE_DURATION = float(getattr(cfg, "INNER_PAUSE_DURATION", 3))
INNER_PAUSE_DURATION_HW = float(getattr(cfg, "INNER_PAUSE_DURATION_HW", 5))
TRAILING_PAUSE_DURATION = float(getattr(cfg, "TRAILING_PAUSE_DURATION", 3))
DEFAULT_FONT_SIZE = int(getattr(cfg, "DEFAULT_FONT_SIZE", 100))
HEADER_GAP_RATIO = float(getattr(cfg, "HEADER_GAP_RATIO", 0.30))
MARGIN_RIGHT = int(getattr(cfg, "MARGIN_RIGHT", 25))
MARGIN_TOP = int(getattr(cfg, "MARGIN_TOP", 15))

PROJECT_MODE = getattr(cfg, "PROJECT_MODE", "Production")


# INTRO_MESSAGES = {
#     "Nufi": "Yū' Mfʉ́ɑ́'sí, Mfāhngə́ə́:",
#     "Swahili": "Sikiliza, rudia na tafsiri:",
#     "Yoruba": "Tẹ́tí gbọ́, tunsọ, ṣe ògbùfọ̀:",
#     "Duala": "Seŋgâ, Timbísɛ́lɛ̂ na Túkwâ:",
# }
# DEFAULT_INTRO = "Listen, repeat and translate:"

INTRO_MESSAGES = cfg.INTRO_MESSAGES

DEFAULT_INTRO = getattr(cfg, "DEFAULT_INTRO", "Listen, repeat and translate:")

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── LOGGING CONFIG ──────────────────────────────────────────────────────

# Use the script name but place it inside the Phrasebook folder
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

logging.info(
    "Text rendering backend: %s",
    "ImageMagick/TextClip" if USE_IMAGEMAGICK else "Pillow (bundled TTF)",
)
logging.info("FONT_PATH_BASIC: %s (exists=%s)", str(BASIC_FONT_PATH), Path(BASIC_FONT_PATH).exists())
logging.info("FONT_PATH_SPECIAL: %s (exists=%s)", str(SPECIAL_FONT_PATH), Path(SPECIAL_FONT_PATH).exists())


_SPECIAL_FONT_NEEDLES = ["ε", "έ", "ɛ̀", "ɛ̄", "ɛ̌", "ɛ̂"]


def _sentence_needs_special_font(sentence: Dict) -> bool:
    parts = [
        str(sentence.get("source", "")),
        str(sentence.get("english", "")),
        str(sentence.get("french", "")),
    ]
    return any(n in part for part in parts for n in _SPECIAL_FONT_NEEDLES)


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

        
# ── PATH SETUP ──────────────────────────────────────────────────────────
# language = LANGUAGE
# mode = MODE

# ── PARSE SENTENCE FILE ─────────────────────────────────────────────────
def load_sentences(txt_file: Path, lang_code: str) -> List[Dict]:
    sentences = []
    with open(txt_file, encoding="utf-8") as file:
        for line in file:
            if "|" not in line:
                continue
            try:
                id_part, english_raw = line.split(")", 1)
                sentence_id = int(id_part.strip())
                _, local, french = [x.strip() for x in line.strip().split("|", 2)]
                english = english_raw.split("|", 1)[0].strip()
                sentences.append({
                    "id": sentence_id,
                    "source": local,
                    "english": english,
                    "french": french,
                    "local_audio": f"{lang_code}_phrasebook_{sentence_id}_padded.mp3",
                    "english_audio": f"english_{sentence_id}.mp3",
                })
            except ValueError:
                print(f"⚠ Invalid line format: {line.strip()}")
    return sentences

# ── CHAPTER DETECTION ───────────────────────────────────────────────────
def get_chapter_ranges(sentences: List[Dict]) -> List[tuple]:
    ranges = []
    start_id = None
    previous_id = None
    for s in sorted(sentences, key=lambda x: x["id"]):
        if s["english"].lower().startswith("chapter"):
            if start_id is not None and previous_id is not None:
                ranges.append((start_id, previous_id))
            start_id = s["id"]
        previous_id = s["id"]
    if start_id and previous_id:
        ranges.append((start_id, previous_id))
    if not ranges and sentences:
        ids = [s["id"] for s in sentences]
        ranges = [(min(ids), max(ids))]
    return ranges

# ── BACKGROUND IMAGE RESIZING ───────────────────────────────────────────
def resize_background(image_path: Path, target_size=VIDEO_RESOLUTION) -> Path:
    with Image.open(image_path) as img:
        if img.size == target_size:
            return image_path
        even_dims = (target_size[0] - target_size[0] % 2, target_size[1] - target_size[1] % 2)
        resized_path = image_path.with_stem(image_path.stem + "_resized")
        img.resize(even_dims, Image.LANCZOS).save(resized_path)
        return resized_path

# ── ATTACH BACKGROUND TO SENTENCES ──────────────────────────────────────
def assign_backgrounds(sentences: List[Dict], backgrounds: List[Path]) -> List[Dict]:
    current_idx = -1
    current_bg = backgrounds[0]
    result = []
    for sentence in sorted(sentences, key=lambda x: x["id"]):
        if sentence["english"].lower().startswith("chapter"):
            current_idx = (current_idx + 1) % len(backgrounds)
            current_bg = backgrounds[current_idx]
        sentence_copy = sentence.copy()
        sentence_copy["background"] = current_bg
        result.append(sentence_copy)
    return result

# ── FONT UTILITY ────────────────────────────────────────────────────────
def calculate_font_size(sentence: Dict) -> int:
    total_chars = len(f"{sentence['source']} {sentence['english']} {sentence['french']}")
    if total_chars < 50:
        return DEFAULT_FONT_SIZE
    elif total_chars < 80:
        return int(DEFAULT_FONT_SIZE * 0.85)
    elif total_chars < 110:
        return int(DEFAULT_FONT_SIZE * 0.75)
    return int(DEFAULT_FONT_SIZE * 0.65)


# Robust text clip creator: Pillow-first (bundled TTF), optional TextClip/ImageMagick
def make_text_clip(text: str, *, font: Path | str, fontsize: int, color: str = "white",
                   size: tuple | None = None, method: str = "caption"):
    # Prefer Pillow by default (bundled TTF), optionally TextClip/ImageMagick
    force_textclip = os.getenv("FORCE_TEXTCLIP", "0") == "1"

    if USE_IMAGEMAGICK or force_textclip:
        try:
            # Try several ways of specifying the font so ImageMagick can resolve it
            font_candidates = []
            try:
                p = Path(font)
                font_candidates.append(str(p))
                font_candidates.append(str(p.resolve()))
                font_candidates.append(p.name)
                font_candidates.append(p.stem)
                # POSIX-style path may help on Windows with ImageMagick
                font_candidates.append(str(p).replace('\\', '/'))
            except Exception:
                font_candidates.append(str(font))

            font_candidates.append(str(font))

            tried = []
            for ftry in font_candidates:
                if ftry in tried:
                    continue
                tried.append(ftry)
                try:
                    tc = TextClip(
                        text,
                        font=ftry,
                        fontsize=int(fontsize),
                        color=color,
                        method=method,
                        size=size if size is not None else None,
                    )
                    if getattr(tc, "w", 0) <= 0 or getattr(tc, "h", 0) <= 0:
                        raise RuntimeError("Empty TextClip created")
                    logging.debug("TextClip created using font spec: %s", ftry)
                    return tc
                except Exception:
                    logging.debug("TextClip font try failed: %s", ftry)
            raise RuntimeError("All TextClip font candidates failed")
        except Exception as e:
            if force_textclip:
                global _WARNED_FORCE_TEXTCLIP_FALLBACK
                if not _WARNED_FORCE_TEXTCLIP_FALLBACK:
                    logging.warning(
                        "FORCE_TEXTCLIP=1 set but TextClip failed; using Pillow fallback instead. (%s)",
                        e,
                    )
                    _WARNED_FORCE_TEXTCLIP_FALLBACK = True
                else:
                    logging.debug(
                        "FORCE_TEXTCLIP=1 set but TextClip failed; using Pillow fallback. (%s)",
                        e,
                    )
            if USE_IMAGEMAGICK:
                logging.debug("TextClip (ImageMagick) failed; falling back to Pillow. (%s)", e)

    logging.debug("Rendering text with Pillow")

    # Pillow fallback (use explicit TTF if available)
    try:
        text = "" if text is None else str(text)
        try:
            pil_font = ImageFont.truetype(str(font), int(fontsize))
        except Exception as e:
            logging.warning("Failed to load TTF font %s; using default font. (%s)", str(font), e)
            pil_font = ImageFont.load_default()

        max_w = None
        if size and size[0]:
            max_w = int(size[0])

        draw_tmp = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
        _, base_line_h = draw_tmp.textsize("Ag", font=pil_font)
        line_gap = max(2, int(int(fontsize) * 0.06))

        def text_width(s: str) -> int:
            return int(draw_tmp.textsize(s, font=pil_font)[0])

        def split_long_token(token: str) -> list[str]:
            # Split an unbreakable token (e.g., aVeryVeryLongWord) into chunks that fit max_w
            if not token:
                return [""]
            parts: list[str] = []
            cur = ""
            for ch in token:
                if cur and text_width(cur + ch) > max_w:
                    parts.append(cur)
                    cur = ch
                else:
                    cur += ch
            if cur:
                parts.append(cur)
            return parts

        def wrap_paragraph(paragraph: str) -> list[str]:
            words = paragraph.split()
            if not words:
                return [""]
            lines: list[str] = []
            cur = ""
            for token in words:
                if not cur:
                    if text_width(token) <= max_w:
                        cur = token
                    else:
                        chunks = split_long_token(token)
                        lines.extend(chunks[:-1])
                        cur = chunks[-1]
                    continue

                test = f"{cur} {token}"
                if text_width(test) <= max_w:
                    cur = test
                else:
                    lines.append(cur)
                    if text_width(token) <= max_w:
                        cur = token
                    else:
                        chunks = split_long_token(token)
                        lines.extend(chunks[:-1])
                        cur = chunks[-1]
            if cur:
                lines.append(cur)
            return lines

        if max_w:
            paragraphs = text.splitlines() or [text]
            lines: list[str] = []
            for para in paragraphs:
                if lines:
                    lines.append("")  # preserve explicit newline between paragraphs
                lines.extend(wrap_paragraph(para))
        else:
            lines = text.splitlines() or [text]

        # Measure text block size
        line_sizes = [draw_tmp.textsize(line, font=pil_font) if line else (0, base_line_h) for line in lines]
        width = max((w for w, h in line_sizes), default=0)
        height = sum((h for w, h in line_sizes), 0) + (len(lines) - 1) * line_gap

        img = Image.new("RGBA", (max(width, 1), max(height, 1)), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        y = 0
        fill = ImageColor.getrgb(color)
        for i, line in enumerate(lines):
            if line:
                draw.text((0, y), line, font=pil_font, fill=fill)
            y += line_sizes[i][1]
            if i != len(lines) - 1:
                y += line_gap

        arr = np.array(img)
        clip = ImageClip(arr).set_duration(0.0)
        clip = clip.set_fps(FRAME_RATE)
        return clip
    except Exception:
        logging.exception("Pillow fallback failed for make_text_clip")
        # final minimal fallback
        img = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
        arr = np.array(img)
        clip = ImageClip(arr).set_duration(0.0)
        clip = clip.set_fps(FRAME_RATE)
        return clip


def fit_caption_block(
    parts: list[tuple[str, str]],
    *,
    font: Path | str,
    fontsize: int,
    max_width: int,
    max_height: int,
    gap: int = 10,
    min_fontsize: int = 18,
) -> tuple[int, list]:
    """
    Create a stacked set of caption clips that fits within max_height by reducing font size.
    Returns (font_size_used, clips) in the same order as parts.
    """
    if not parts:
        return int(fontsize), []

    fs = int(fontsize)
    min_fs = max(8, int(min_fontsize))
    max_width_i = int(max_width)
    max_height_i = int(max_height)

    while True:
        clips = [
            make_text_clip(
                text,
                font=font,
                fontsize=fs,
                color=color,
                method="caption",
                size=(max_width_i, None),
            )
            for text, color in parts
        ]
        total_h = sum(int(getattr(c, "h", 0) or 0) for c in clips) + gap * (len(clips) - 1)
        if max_height_i <= 0 or total_h <= max_height_i or fs <= min_fs:
            return fs, clips

        new_fs = max(min_fs, int(fs * 0.9))
        fs = new_fs if new_fs < fs else (fs - 1)

# ── BUILD SINGLE VIDEO ──────────────────────────────────────────────────

def create_video_clip(sentence: Dict):
    """
    Build a single video clip for one sentence.
    Uses dynamic heights so text never overlaps when wrapping.
    """

    print(f"▶ Creating video for sentence ID {sentence['id']}")
    output_path = VIDEO_OUT_DIR / f"{LANGUAGE.lower()}_sentence_{sentence['id']}.mp4"

    if output_path.exists() and not REBUILD_ALL:
        print(f"✔ Skipped (already exists): {output_path.name}")
        return

    local_audio_path = LOCAL_AUDIO_DIR / sentence["local_audio"]
    english_audio_path = ENG_AUDIO_DIR / sentence["english_audio"]
    if not local_audio_path.exists():
        print(f"⚠ Missing local audio for sentence {sentence['id']}")
        return

    local_audio = AudioFileClip(str(local_audio_path))
    english_missing = not english_audio_path.exists()
    if english_missing:
        print(f"⚠ Missing English audio for sentence {sentence['id']}; using local only")
        english_audio = local_audio
    else:
        english_audio = AudioFileClip(str(english_audio_path))

    # --- Audio sequencing ---
    if MODE.lower() == "lecture":
        if english_missing:
            total_duration = local_audio.duration + TRAILING_PAUSE_DURATION
            audio = CompositeAudioClip([
                local_audio.set_start(0),
            ])
        else:
            total_duration = english_audio.duration + INNER_PAUSE_DURATION + local_audio.duration + TRAILING_PAUSE_DURATION
            audio = CompositeAudioClip([
                english_audio.set_start(0),
                local_audio.set_start(english_audio.duration + INNER_PAUSE_DURATION),
            ])
        
        #Preview audio
        # audio.fps = 44100 
        # audio.preview()   # plays in real time using pygame

    else:  # homework mode
        local_audio_duration, pause = local_audio.duration, INNER_PAUSE_DURATION_HW
        local_lang_1_start = 0
        local_lang_2_start = local_audio_duration + pause
        local_lang_3_start = local_lang_2_start + local_audio_duration + pause
        eng_start = local_lang_3_start + local_audio_duration + pause
        total_duration = eng_start + english_audio.duration + TRAILING_PAUSE_DURATION

        audio = CompositeAudioClip([
            local_audio.set_start(local_lang_1_start),
            local_audio.set_start(local_lang_2_start),
            local_audio.set_start(local_lang_3_start),
            english_audio.set_start(eng_start),
        ])
        #Preview audio
        # audio.fps = 44100 
        # audio.preview()   # plays in real time using pygame

    # Ensure audio spans the full video duration (pads with silence at the end).
    audio = audio.set_duration(total_duration)

    # --- Font & spacing ---
    font_size = calculate_font_size(sentence)
    font_path = SPECIAL_FONT_PATH if _sentence_needs_special_font(sentence) else BASIC_FONT_PATH
    y_position_lecture = 120
    y_position_homework = 120

    bg = ImageClip(str(sentence["background"])).set_duration(total_duration)
    clips = [bg]

    # --- Header row ---
    lang_clip = make_text_clip(LANGUAGE.title(), font=font_path,
                               fontsize=int(font_size * 0.55), color="yellow", method="label")
    lang_clip = lang_clip.set_position((15, MARGIN_TOP)).set_duration(total_duration)
    clips.append(lang_clip)

    support_clip = make_text_clip("Please Support Resulam", font=font_path,
                                 fontsize=int(font_size * 0.5), color="yellow", method="label")
    support_clip = support_clip.set_position(("right", MARGIN_TOP)).set_duration(total_duration)
    clips.append(support_clip)
    # Use make_text_clip for the numeric badge so fallback works if TextClip can't render
    num_clip = make_text_clip(str(sentence["id"]), font=font_path,
                              fontsize=int(font_size * 0.6), color="white", method="label")
    num_clip = num_clip.set_position(
        (VIDEO_RESOLUTION[0] - num_clip.w - MARGIN_RIGHT, MARGIN_TOP + support_clip.h + 10)
    ).set_duration(total_duration)
    clips.append(num_clip)

    # --- Captions ---
    if MODE.lower() == "lecture":
        current_y = y_position_lecture
        max_width = VIDEO_RESOLUTION[0] - 200
        bottom_safe = 140  # reserve space for logos + padding
        max_block_h = VIDEO_RESOLUTION[1] - current_y - bottom_safe
        min_caption_fs = int(getattr(cfg, "MIN_CAPTION_FONT_SIZE", 18))

        _, caption_clips = fit_caption_block(
            [
                (sentence["source"], "white"),
                (sentence["english"], "yellow"),
                (sentence["french"], "white"),
            ],
            font=font_path,
            fontsize=font_size,
            max_width=max_width,
            max_height=max_block_h,
            gap=10,
            min_fontsize=min_caption_fs,
        )
        src_clip, eng_clip, fr_clip = caption_clips

        src_clip = src_clip.set_duration(total_duration).set_position(("center", current_y))
        clips.append(src_clip)
        current_y += src_clip.h + 10

        eng_clip = eng_clip.set_duration(total_duration).set_position(("center", current_y))
        clips.append(eng_clip)
        current_y += eng_clip.h + 10

        fr_clip = fr_clip.set_duration(total_duration).set_position(("center", current_y))
        clips.append(fr_clip)

    else:  # homework
        intro_msg = INTRO_MESSAGES.get(LANGUAGE.title(), DEFAULT_INTRO)
        current_y = y_position_homework

        # Intro
        intro_clip = make_text_clip(intro_msg, font=font_path, fontsize=int(font_size*0.9),
                         color="white", method="label")
        intro_clip = intro_clip.set_position(("center", current_y)).set_start(0).set_duration(local_lang_2_start)
        clips.append(intro_clip)
        current_y += intro_clip.h + 10

        repeat_clip = make_text_clip("Listen, repeat and translate", font=font_path, fontsize=font_size,
                         color="yellow", method="label")
        repeat_clip = repeat_clip.set_position(("center", current_y)).set_start(0).set_duration(local_lang_2_start)
        clips.append(repeat_clip)

        # Fit caption font size so the (local + English + French) stack always stays on-screen.
        max_width = VIDEO_RESOLUTION[0] - 200
        bottom_safe = 140  # reserve space for logos + padding
        max_block_h = VIDEO_RESOLUTION[1] - y_position_homework - bottom_safe
        min_caption_fs = int(getattr(cfg, "MIN_CAPTION_FONT_SIZE", 18))
        caption_fs, caption_clips = fit_caption_block(
            [
                (sentence["source"], "white"),
                (sentence["english"], "yellow"),
                (sentence["french"], "white"),
            ],
            font=font_path,
            fontsize=font_size,
            max_width=max_width,
            max_height=max_block_h,
            gap=10,
            min_fontsize=min_caption_fs,
        )
        src_tmp, eng_tmp, fr_tmp = caption_clips

        # Second playback → only local
        src_clip2 = make_text_clip(sentence["source"], font=font_path, fontsize=caption_fs,
                       color="white", method="caption", size=(VIDEO_RESOLUTION[0]-200, None))
        src_clip2 = src_clip2.set_position(("center", y_position_homework)).set_start(local_lang_2_start).set_duration(local_lang_3_start - local_lang_2_start)
        clips.append(src_clip2)

        # Third playback → dynamic stack
        current_y = y_position_homework
        src_clip3 = src_tmp.set_start(local_lang_3_start).set_duration(total_duration - local_lang_3_start)
        src_clip3 = src_clip3.set_position(("center", current_y))
        clips.append(src_clip3)
        current_y += src_clip3.h + 10

        eng_clip3 = eng_tmp.set_start(local_lang_3_start).set_duration(total_duration - local_lang_3_start)
        eng_clip3 = eng_clip3.set_position(("center", current_y))
        clips.append(eng_clip3)
        current_y += eng_clip3.h + 10

        fr_clip3 = fr_tmp.set_start(local_lang_3_start).set_duration(total_duration - local_lang_3_start)
        fr_clip3 = fr_clip3.set_position(("center", current_y))
        clips.append(fr_clip3)

    # --- Logos ---
    for side in ["left", "right"]:
        clips.append(ImageClip(str(LOGO_PATH)).resize(height=100)
                     .set_position((side, "bottom")).set_duration(total_duration))

    # --- Render ---
    temp_audio_file = f"temp_audio_{uuid4().hex}.m4a"
    temp_video_file = output_path.with_suffix(".tmp.mp4")
    final = None
    try:
        final = CompositeVideoClip(clips).set_audio(audio)
        final.write_videofile(
            str(temp_video_file), fps=FRAME_RATE, codec="libx264", audio_codec="aac",
            temp_audiofile=temp_audio_file, remove_temp=True,
            audio_bitrate="192k", audio_fps=AUDIO_SAMPLE_RATE,
            ffmpeg_params=[
                "-pix_fmt", "yuv420p",
                "-profile:v", "high",
                "-level", "4.1",
                "-movflags", "+faststart",
            ],
            preset=str(getattr(cfg, "X264_PRESET", "superfast")), threads=FFMPEG_THREADS,
        )

        # Close MoviePy objects before remux/rename on Windows.
        try:
            final.close()
        except Exception:
            pass
        final = None

        # Optional remux step (copy video, re-encode audio) to maximize WMP compatibility.
        if POSTPROCESS_MP4 and shutil.which("ffmpeg"):
            temp_muxed_file = output_path.with_suffix(".mux.mp4")
            cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_video_file),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", str(AUDIO_SAMPLE_RATE),
                "-ac", "2",
                "-movflags", "+faststart",
                "-brand", "mp42",
                str(temp_muxed_file),
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if proc.returncode == 0 and temp_muxed_file.exists():
                temp_video_file.unlink(missing_ok=True)
                temp_video_file = temp_muxed_file
            else:
                try:
                    temp_muxed_file.unlink(missing_ok=True)
                except Exception:
                    pass

        # Give the OS a moment to release file handles on Windows before rename.
        for attempt in range(6):
            try:
                time.sleep(0.15 * attempt)
                temp_video_file.replace(output_path)
                break
            except PermissionError:
                if attempt == 5:
                    raise

        print(f"✅ Rendered: {output_path.name}")
    except Exception as error:
        print(f"❌ Error rendering sentence {sentence['id']}: {error}")
        Path(temp_audio_file).unlink(missing_ok=True)
        Path(temp_video_file).unlink(missing_ok=True)
    finally:
        try:
            if final is not None:
                final.close()
        except Exception:
            pass
        try:
            audio.close()
        except Exception:
            pass
        try:
            local_audio.close()
        except Exception:
            pass
        try:
            english_audio.close()
        except Exception:
            pass
        for c in clips:
            try:
                if hasattr(c, "close"):
                    c.close()
            except Exception:
                pass


# ── MULTI-THREAD RENDERING ──────────────────────────────────────────────
# chapter_ranges = chapters
# sentences = tagged_sentences
# paths = CONFIG
# mode = MODE 
# chapter_index, start_id, end_id = 1, 1, 173
# sentence = segment[0]

def render_all_sentences(
    sentences: List[Dict],
    chapter_ranges: List[tuple],
    *,
    start_chapter: int | None = None,
    end_chapter: int | None = None,
    start_sentence: int | None = None,
    end_sentence: int | None = None,
):
    """
    Render sentences into videos by chapter range or by sentence-ID range.

    Priority:
      • If start_sentence and end_sentence are provided → render by sentence IDs.
      • Else → render by chapter ranges.

    NOTE: This function always uses the original sentence["id"] from the text file,
    so numbers in the video header stay consistent (e.g., 1638, 1639…).
    """
    # Use ThreadPoolExecutor to control parallel renders and avoid manual semaphore logic
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def worker(sentence: Dict):
        try:
            create_video_clip(sentence)
        except Exception as e:
            # Include sentence id in the error log and print traceback for debugging
            logging.error(f"❌ Error rendering sentence {sentence.get('id')}: {e}")
            import traceback as _tb
            _tb.print_exc()
            # Re-raise so the caller can observe the failure if needed
            raise

    selected_ids = getattr(cfg, "SELECTED_SENTENCE_IDS", None)
    if selected_ids:
        selected = [s for s in sentences if s.get("id") in selected_ids]
        selected.sort(key=lambda x: x["id"])
        if not selected:
            print("⚠ No sentences found for SELECTED_SENTENCE_IDS. Nothing to render.")
            return
        print(f"▶ Rendering {len(selected)} selected sentence(s): {selected[0]['id']}…{selected[-1]['id']}")
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as ex:
            futures = [ex.submit(worker, s) for s in selected]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logging.error(f"❌ Error rendering: {e}")
        print("✔ Finished rendering selected sentences")
        return

    # --- Option A: sentence range ---
    if start_sentence is not None or end_sentence is not None:
        if start_sentence is None or end_sentence is None:
            raise ValueError("Both start_sentence and end_sentence must be provided together.")
        if start_sentence > end_sentence:
            raise ValueError(
                f"Invalid sentence range: start_sentence ({start_sentence}) "
                f"cannot be greater than end_sentence ({end_sentence})."
            )

        selected = [s for s in sentences if start_sentence <= s["id"] <= end_sentence]
        if not selected:
            print(f"⚠ No sentences found in range {start_sentence}–{end_sentence}. Nothing to render.")
            return

        # Non-destructive warning: old clips outside the requested range can confuse later steps.
        try:
            import re as _re
            existing_ids: list[int] = []
            for p in VIDEO_OUT_DIR.glob(f"{LANGUAGE.lower()}_sentence_*.mp4"):
                m = _re.search(r"(?:^|_)sentence_(\d+)$", p.stem, _re.IGNORECASE)
                if not m:
                    continue
                existing_ids.append(int(m.group(1)))
            outside = sorted({i for i in existing_ids if i < start_sentence or i > end_sentence})
            if outside:
                print(
                    f"⚠ Output folder already has {len(outside)} clip(s) outside {start_sentence}–{end_sentence} "
                    f"(e.g. {outside[:10]}). Step 3/5 may include them unless you filter chapters or clean old outputs."
                )
        except Exception:
            pass

        print(f"▶ Rendering {len(selected)} sentence(s) in range {start_sentence}–{end_sentence}…")
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as ex:
            futures = [ex.submit(worker, s) for s in selected]
            for fut in as_completed(futures):
                try:
                    fut.result()
                except Exception as e:
                    logging.error(f"❌ Error rendering: {e}")
        print(f"✔ Finished rendering sentences {start_sentence}–{end_sentence}")
        return

    # --- Option B: chapter range ---
    if start_chapter is None:
        start_chapter = 1
    if end_chapter is None:
        end_chapter = len(chapter_ranges)
    if start_chapter > end_chapter:
        raise ValueError(
            f"Invalid chapter range: start_chapter ({start_chapter}) "
            f"cannot be greater than end_chapter ({end_chapter})."
        )

    # Collect futures to wait on
    all_futures = []
    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as ex:
        for chapter_index, (start_id, end_id)  in enumerate(chapter_ranges, start=1):
            if chapter_index < start_chapter:
                continue
            if chapter_index > end_chapter:
                break

            segment = [s for s in sentences if start_id <= s["id"] <= end_id]
            if not segment:
                print(f"⚠ Chapter {chapter_index} has no sentences in the parsed data; skipping.")
                continue

            print(f"▶ Processing Chapter {chapter_index} ({start_id}–{end_id}) with {len(segment)} sentence(s)…")

            for sentence in segment:
                all_futures.append(ex.submit(worker, sentence))

        # Wait for all submitted tasks and report errors as they occur
        for fut in as_completed(all_futures):
            try:
                fut.result()
            except Exception as e:
                logging.error(f"❌ Error rendering: {e}")

    print(f"✔ Finished rendering Chapters {start_chapter}–{end_chapter} for {LANGUAGE.title()} [{MODE}]")

# ── ENTRY POINT ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    # start_chapter=1
    # end_chapter=32
    
    # start_sentence, end_sentence=1638, 1638
    # start_sentence, end_sentence=None, None
    
    def _maybe_int(v):
        if v is None or v == "":
            return None
        try:
            return int(v)
        except Exception:
            return None

    start_chapter = _maybe_int(getattr(cfg, "START_CHAPTER", None))
    end_chapter = _maybe_int(getattr(cfg, "END_CHAPTER", None))
    start_sentence = _maybe_int(getattr(cfg, "START_SENTENCE", None))
    end_sentence = _maybe_int(getattr(cfg, "END_SENTENCE", None))

    # Optional env override (useful when running step0_main_pipeline from different configs)
    env_start_sentence = os.getenv("START_SENTENCE") or os.getenv("START_ID")
    env_end_sentence = os.getenv("END_SENTENCE") or os.getenv("END_ID")
    if env_start_sentence is not None or env_end_sentence is not None:
        env_start_sentence_i = _maybe_int(env_start_sentence)
        env_end_sentence_i = _maybe_int(env_end_sentence)
        if env_start_sentence_i is None or env_end_sentence_i is None:
            logging.warning(
                "START_SENTENCE/END_SENTENCE env override ignored: both must be set to integers."
            )
        else:
            start_sentence, end_sentence = env_start_sentence_i, env_end_sentence_i

    logging.info(
        f"Filters: START_SENTENCE={start_sentence} END_SENTENCE={end_sentence} "
        f"START_CHAPTER={start_chapter} END_CHAPTER={end_chapter} "
        f"(cfg={getattr(cfg, '__file__', '?')}, cwd={Path.cwd()})"
    )


    
    with log_time("Load Sentences"):
        raw_sentences = load_sentences(SENTENCES_PATH, LANGUAGE.lower())

    with log_time("Detect Chapters"):
        chapters = get_chapter_ranges(raw_sentences)

    with log_time("Prepare Backgrounds"):
        backgrounds = [resize_background(p) for ext in ("*.png", "*.jpg", "*.jpeg")
                    for p in BACKGROUND_DIR.glob(ext)]
       
        
        if not BACKGROUND_DIR.exists() or not any(BACKGROUND_DIR.glob("*")):
            BACKGROUND_DIR = get_asset_path(f"Backgrounds_Selected")
            backgrounds = [resize_background(p) for ext in ("*.png", "*.jpg", "*.jpeg")
                        for p in BACKGROUND_DIR.glob(ext)]

    with log_time("Assign Backgrounds"):
        tagged_sentences = assign_backgrounds(raw_sentences, backgrounds)

   
    with log_time("Render Sentences"):
        render_all_sentences(
            tagged_sentences, chapters,
            start_chapter=start_chapter, 
            end_chapter=end_chapter,
            start_sentence=start_sentence, 
            end_sentence=end_sentence, 
        )
