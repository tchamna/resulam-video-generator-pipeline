
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
from pathlib import Path


# ── USER SETTINGS ───────────────────────────────────────────────────────
LANGUAGE = "Duala"          # e.g., "Nufi", "Yoruba", "Duala"
MODE = "lecture"            # "lecture" or "homework"
# MODE = "homework"            # "lecture" or "homework"
REBUILD_ALL = False         # Force regeneration of existing videos
REBUILD_ALL = True         # Force regeneration of existing videos

# ── PARALLELISM CONFIG ──────────────────────────────────────────────────
FFMPEG_THREADS = 4
MAX_PARALLEL_JOBS = 5
AVAILABLE_CPUS = os.cpu_count() or 1

while FFMPEG_THREADS > AVAILABLE_CPUS:
    FFMPEG_THREADS = max(1, FFMPEG_THREADS // 2)
MAX_PARALLEL_JOBS = min(MAX_PARALLEL_JOBS, AVAILABLE_CPUS // FFMPEG_THREADS)

# ── PATHS SETUP ─────────────────────────────────────────────────────
BASE_DIR = Path(os.getcwd())
FONT_PATH = BASE_DIR / "Assets"/ "Fonts" / "arialbd.ttf"
LOGO_PATH = BASE_DIR/ "Assets" / "resulam_logo_resurrectionLangue.png"

# ── FONTS SETUP ─────────────────────────────────────────────────────

VIDEO_RESOLUTION = (1920, 1080)
FRAME_RATE = 24
REPEAT_PAUSE_SECONDS = 5
TRAILING_PAUSE_SECONDS = 3
DEFAULT_FONT_SIZE = 100
HEADER_GAP_RATIO = 0.30
MARGIN_RIGHT = 25
MARGIN_TOP = 15

PROJECT_MODE = "Test"  # or "Production"
PROJECT_MODE = "Production"  # or "Production"

INTRO_MESSAGES = {
    "Nufi": "Yū' Mfʉ́ɑ́'sí, Mfāhngə́ə́:",
    "Swahili": "Sikiliza, rudia na tafsiri:",
    "Yoruba": "Tẹ́tí gbọ́, tunsọ, ṣe ògbùfọ̀:",
    "Duala": "Seŋgâ, Timbísɛ́lɛ̂ na Túkwâ:",
}
DEFAULT_INTRO = "Listen, repeat and translate:"
ImageFile.LOAD_TRUNCATED_IMAGES = True



# ── LOGGING CONFIG ──────────────────────────────────────────────────────
log_file = Path(__file__).with_suffix(".log")  # saves alongside your script
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),  # write to file
        logging.StreamHandler()  # also keep printing to console
    ]
)

@contextmanager
def log_time(step_name: str):
    """Context manager to log execution time of a code block."""
    start = time.perf_counter()
    logging.info(f"▶️ Starting {step_name}...")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"⏱ Finished {step_name} in {elapsed:.2f} sec")

# ── PATH SETUP ──────────────────────────────────────────────────────────
# language = LANGUAGE
# mode = MODE




def get_project_paths(language: str, mode: str):
    
    local_language_title = language.title()
    language_lower = language.lower()
    mode_folder = "Lecture" if mode.lower() == "lecture" else "Homework"

    assets_dir = BASE_DIR / "Assets"

    # Output directory (always created)
    output_dir = assets_dir / "Languages" / f"{local_language_title}Phrasebook" / "Results_Videos" / mode_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    # Backgrounds: fallback to "Default" if language folder doesn't exist
    background_dir = assets_dir / "Backgrounds_Selected" / local_language_title
    if not background_dir.exists():
        background_dir = background_dir.parent / "Default"
    # Local audio: depends on PROJECT_MODE
    suffix = "OnlyTest" if PROJECT_MODE.lower() == "test" else "Only"
    # local_audio_dir = assets_dir / "Languages" / f"{local_language_title}Phrasebook" / f"{local_language_title}{suffix}" / "gen2_normalized_padded"
    local_audio_dir = BASE_DIR /"Assets" / "Languages" / f"{local_language_title}Phrasebook" / "Results_Audios" / "gen2_normalized_padded"

    return {
        "language": local_language_title,
        "lang_lower": language_lower,
        "background_dir": background_dir,
        "local_audio_dir": local_audio_dir,
        "english_audio_dir": BASE_DIR / "EnglishOnly",
        "output_dir": output_dir,
        "sentence_file": assets_dir / "Languages" / f"{local_language_title}Phrasebook" / f"{language_lower}_english_french_phrasebook_sentences_list.txt",
    }

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

def create_video_clip(sentence: Dict, paths: Dict, mode: str):
    """
    Build a single video clip for one sentence.
    Shows correct sentence ID in header and captions (source, english, french).
    """

    print(f"▶ Creating video for sentence ID {sentence['id']}")

    output_path = paths["output_dir"] / f"{paths['lang_lower']}_sentence_{sentence['id']}.mp4"
    if output_path.exists() and not REBUILD_ALL:
        print(f"✔ Skipped (already exists): {output_path.name}")
        return

    local_audio_path = paths["local_audio_dir"] / sentence["local_audio"]
    english_audio_path = BASE_DIR/"Assets"/"EnglishOnly" / sentence["english_audio"]

    if not local_audio_path.exists() or not english_audio_path.exists():
        print(f"⚠ Missing audio for sentence {sentence['id']}")
        return

    local_audio = AudioFileClip(str(local_audio_path))
    english_audio = AudioFileClip(str(english_audio_path))

    # --- Audio sequencing ---
    if mode == "lecture":
        total_duration = english_audio.duration + REPEAT_PAUSE_SECONDS + local_audio.duration + TRAILING_PAUSE_SECONDS
        audio = CompositeAudioClip([
            english_audio.set_start(0),
            local_audio.set_start(english_audio.duration + REPEAT_PAUSE_SECONDS)
        ])
    else:  # homework mode
        d = local_audio.duration
        pause = REPEAT_PAUSE_SECONDS
        nufi_1_start = 0
        nufi_2_start = d + pause
        nufi_3_start = nufi_2_start + d + pause
        eng_start = nufi_3_start + d + pause
        total_duration = eng_start + english_audio.duration + TRAILING_PAUSE_SECONDS

        audio = CompositeAudioClip([
            local_audio.set_start(nufi_1_start),
            local_audio.set_start(nufi_2_start),
            local_audio.set_start(nufi_3_start),
            english_audio.set_start(eng_start),
        ])

    font_size = calculate_font_size(sentence)
    clips = [ImageClip(str(sentence["background"])).set_duration(total_duration)]

    def add_text(text, color, start, duration, y_offset, wrap=True):
        size = (VIDEO_RESOLUTION[0] - 200, None) if wrap else None
        clip = TextClip(text, font=str(FONT_PATH), fontsize=font_size,
                        color=color, method="caption" if wrap else "label", size=size)
        clip = clip.set_position(("center", y_offset)).set_start(start).set_duration(duration)
        clips.append(clip)
        return y_offset + clip.h + int(font_size * 0.2)

    # --- Header row ---
    y_header = MARGIN_TOP
    clips.append(TextClip(paths["language"], font=str(FONT_PATH),
                          fontsize=int(font_size * 0.55), color="yellow", method="label")
                 .set_position((15, y_header)).set_duration(total_duration))

    support_clip = TextClip("Please Support Resulam", font=str(FONT_PATH),
                            fontsize=int(font_size * 0.5), color="yellow", method="label")
    support_clip = support_clip.set_position(("right", y_header)).set_duration(total_duration)
    clips.append(support_clip)

    # ✅ Correct sentence number
    sentence_number = str(sentence["id"])
    num_clip = TextClip(sentence_number, font=str(FONT_PATH),
                        fontsize=int(font_size * 0.6), color="white", method="label")
    num_clip = num_clip.set_position(
        (VIDEO_RESOLUTION[0] - num_clip.w - MARGIN_RIGHT,
         y_header + support_clip.h + int(font_size * 0.15))
    ).set_duration(total_duration)
    clips.append(num_clip)

    # --- Captions ---
    y_text = y_header + int(font_size * HEADER_GAP_RATIO) + 80
    if mode == "lecture":
        y_text = add_text(sentence["source"], "white", 0, total_duration, y_text)
        y_text = add_text(sentence["english"], "yellow", 0, total_duration, y_text)
        _ = add_text(sentence["french"], "white", 0, total_duration, y_text)
    else:  # homework mode
        intro_msg = INTRO_MESSAGES.get(paths["language"], DEFAULT_INTRO)

        # 1. First Nufi playback → intro only
        y_text = add_text(intro_msg, "white", 0, nufi_2_start, y_text)
        y_text = add_text("Listen, repeat and translate", "yellow", 0, nufi_2_start, y_text)

        # 2. Second Nufi playback → only local sentence
        y_text = add_text(sentence["source"], "white", nufi_2_start, nufi_3_start - nufi_2_start, y_text)

        # 3. Third Nufi playback → local + English + French
        y_text = add_text(sentence["source"], "white", nufi_3_start, total_duration - nufi_3_start, y_text)
        y_text = add_text(sentence["english"], "yellow", nufi_3_start, total_duration - nufi_3_start, y_text)
        _ = add_text(sentence["french"], "white", nufi_3_start, total_duration - nufi_3_start, y_text)

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
# paths = config
# mode = MODE 

def render_all_sentences(
    sentences: List[Dict],
    chapter_ranges: List[tuple],
    paths: Dict,
    mode: str,
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
    semaphore = threading.Semaphore(MAX_PARALLEL_JOBS)

    def worker(sentence: Dict):
        with semaphore:
            create_video_clip(sentence, paths, mode)

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
        threads = []
        for sentence in selected:
            t = threading.Thread(target=worker, args=(sentence,))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
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

    threads: list[threading.Thread] = []
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
            t = threading.Thread(target=worker, args=(sentence,))
            t.start()
            threads.append(t)

    for t in threads:
        t.join()

    print(f"✔ Finished rendering Chapters {start_chapter}–{end_chapter} for {paths['language']} [{mode}]")

# ── ENTRY POINT ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    
    start_chapter=1
    end_chapter=32
    
    start_sentence, end_sentence=1638, 1638
    start_sentence, end_sentence=None, None

    # config = get_project_paths(LANGUAGE, MODE)
    # raw_sentences = load_sentences(config["sentence_file"], 
    #                                config["lang_lower"])
    # chapters = get_chapter_ranges(raw_sentences)
    # backgrounds = [resize_background(p) for ext in ("*.png", "*.jpg", "*.jpeg") for p in config["background_dir"].glob(ext)]
    # tagged_sentences = assign_backgrounds(raw_sentences, backgrounds)
    
    # render_all_sentences(tagged_sentences, 
    #                      chapters, 
    #                      config, 
    #                      MODE,
    #                      start_chapter=start_chapter,
    #                      end_chapter=end_chapter
    #                      )
    
    # if __name__ == "__main__":
    with log_time("Project Path Setup"):
        config = get_project_paths(LANGUAGE, MODE)

    with log_time("Load Sentences"):
        raw_sentences = load_sentences(config["sentence_file"], config["lang_lower"])

    with log_time("Detect Chapters"):
        chapters = get_chapter_ranges(raw_sentences)

    with log_time("Prepare Backgrounds"):
        backgrounds = [resize_background(p) for ext in ("*.png", "*.jpg", "*.jpeg")
                       for p in config["background_dir"].glob(ext)]

    with log_time("Assign Backgrounds"):
        tagged_sentences = assign_backgrounds(raw_sentences, backgrounds)

    with log_time("Render Sentences"):
        render_all_sentences(tagged_sentences, chapters, config, MODE,
                             start_chapter=1, end_chapter=32)

