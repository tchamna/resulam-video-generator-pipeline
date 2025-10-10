
from __future__ import annotations
import os
import shutil
import threading
from pathlib import Path
from uuid import uuid4
from typing import List, Dict
from PIL import Image, ImageFile
from moviepy.editor import (
    AudioFileClip, CompositeAudioClip, CompositeVideoClip,
    ImageClip, TextClip,
)


import time
import logging
from contextlib import contextmanager

import step0_config as cfg



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

FONT_PATH = get_asset_path( cfg.FONT_PATH)

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
    if not local_audio_path.exists() or not english_audio_path.exists():
        print(f"⚠ Missing audio for sentence {sentence['id']}")
        return

    local_audio = AudioFileClip(str(local_audio_path))
    english_audio = AudioFileClip(str(english_audio_path))

    # --- Audio sequencing ---
    if MODE.lower() == "lecture":
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

    # --- Font & spacing ---
    font_size = calculate_font_size(sentence)
    y_position_lecture = 120
    y_position_homework = 120

    bg = ImageClip(str(sentence["background"])).set_duration(total_duration)
    clips = [bg]

    # --- Header row ---
    clips.append(TextClip(LANGUAGE.title(), font=str(FONT_PATH),
                          fontsize=int(font_size * 0.55), color="yellow", method="label")
                 .set_position((15, MARGIN_TOP)).set_duration(total_duration))
    support_clip = TextClip("Please Support Resulam", font=str(FONT_PATH),
                            fontsize=int(font_size * 0.5), color="yellow", method="label")
    support_clip = support_clip.set_position(("right", MARGIN_TOP)).set_duration(total_duration)
    clips.append(support_clip)
    num_clip = TextClip(str(sentence["id"]), font=str(FONT_PATH),
                        fontsize=int(font_size * 0.6), color="white", method="label")
    num_clip = num_clip.set_position(
        (VIDEO_RESOLUTION[0] - num_clip.w - MARGIN_RIGHT, MARGIN_TOP + support_clip.h + 10)
    ).set_duration(total_duration)
    clips.append(num_clip)

    # --- Captions ---
    if MODE.lower() == "lecture":
        current_y = y_position_lecture
        # Local
        src_clip = TextClip(sentence["source"], font=str(FONT_PATH), fontsize=font_size,
                            color="white", method="caption", size=(VIDEO_RESOLUTION[0]-200, None)
                            ).set_duration(total_duration)
        src_clip = src_clip.set_position(("center", current_y))
        clips.append(src_clip)
        current_y += src_clip.h + 10

        # English
        eng_clip = TextClip(sentence["english"], font=str(FONT_PATH), fontsize=font_size,
                            color="yellow", method="caption", size=(VIDEO_RESOLUTION[0]-200, None)
                            ).set_duration(total_duration)
        eng_clip = eng_clip.set_position(("center", current_y))
        clips.append(eng_clip)
        current_y += eng_clip.h + 10

        # French
        fr_clip = TextClip(sentence["french"], font=str(FONT_PATH), fontsize=font_size,
                           color="white", method="caption", size=(VIDEO_RESOLUTION[0]-200, None)
                           ).set_duration(total_duration)
        fr_clip = fr_clip.set_position(("center", current_y))
        clips.append(fr_clip)

    else:  # homework
        intro_msg = INTRO_MESSAGES.get(LANGUAGE.title(), DEFAULT_INTRO)
        current_y = y_position_homework

        # Intro
        intro_clip = TextClip(intro_msg, font=str(FONT_PATH), fontsize=int(font_size*0.9),
                              color="white", method="label"
                              ).set_position(("center", current_y)).set_start(0).set_duration(local_lang_2_start)
        clips.append(intro_clip)
        current_y += intro_clip.h + 10

        repeat_clip = TextClip("Listen, repeat and translate", font=str(FONT_PATH), fontsize=font_size,
                               color="yellow", method="label"
                               ).set_position(("center", current_y)).set_start(0).set_duration(local_lang_2_start)
        clips.append(repeat_clip)

        # Second playback → only local
        src_clip2 = TextClip(sentence["source"], font=str(FONT_PATH), fontsize=font_size,
                             color="white", method="caption", size=(VIDEO_RESOLUTION[0]-200, None)
                             ).set_position(("center", y_position_homework)
                             ).set_start(local_lang_2_start).set_duration(local_lang_3_start - local_lang_2_start)
        clips.append(src_clip2)

        # Third playback → dynamic stack
        current_y = y_position_homework
        src_clip3 = TextClip(sentence["source"], font=str(FONT_PATH), fontsize=font_size,
                             color="white", method="caption", size=(VIDEO_RESOLUTION[0]-200, None)
                             ).set_start(local_lang_3_start).set_duration(total_duration - local_lang_3_start)
        src_clip3 = src_clip3.set_position(("center", current_y))
        clips.append(src_clip3)
        current_y += src_clip3.h + 10

        eng_clip3 = TextClip(sentence["english"], font=str(FONT_PATH), fontsize=font_size,
                             color="yellow", method="caption", size=(VIDEO_RESOLUTION[0]-200, None)
                             ).set_start(local_lang_3_start).set_duration(total_duration - local_lang_3_start)
        eng_clip3 = eng_clip3.set_position(("center", current_y))
        clips.append(eng_clip3)
        current_y += eng_clip3.h + 10

        fr_clip3 = TextClip(sentence["french"], font=str(FONT_PATH), fontsize=font_size,
                            color="white", method="caption", size=(VIDEO_RESOLUTION[0]-200, None)
                            ).set_start(local_lang_3_start).set_duration(total_duration - local_lang_3_start)
        fr_clip3 = fr_clip3.set_position(("center", current_y))
        clips.append(fr_clip3)

    # --- Logos ---
    for side in ["left", "right"]:
        clips.append(ImageClip(str(LOGO_PATH)).resize(height=100)
                     .set_position((side, "bottom")).set_duration(total_duration))

    # --- Render ---
    temp_audio_file = f"temp_audio_{uuid4().hex}.m4a"
    temp_video_file = output_path.with_suffix(".tmp.mp4")
    try:
        CompositeVideoClip(clips).set_audio(audio).write_videofile(
            str(temp_video_file), fps=FRAME_RATE, codec="libx264", audio_codec="aac",
            temp_audiofile=temp_audio_file, remove_temp=True,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-profile:v", "high", "-level", "4.1", "-movflags", "+faststart"],
            preset="ultrafast", threads=FFMPEG_THREADS,
        )
        shutil.move(temp_video_file, output_path)
        print(f"✅ Rendered: {output_path.name}")
    except Exception as error:
        print(f"❌ Error rendering sentence {sentence['id']}: {error}")
        Path(temp_audio_file).unlink(missing_ok=True)
        Path(temp_video_file).unlink(missing_ok=True)


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
        create_video_clip(sentence)

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
    
    start_chapter = getattr(cfg, "START_CHAPTER", None)
    end_chapter   = getattr(cfg, "END_CHAPTER", None)
    start_sentence = getattr(cfg, "START_SENTENCE", None)
    end_sentence   = getattr(cfg, "END_SENTENCE", None)


    
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
