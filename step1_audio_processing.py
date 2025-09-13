
import os
import shutil
from natsort import natsorted
from pathlib import Path

import glob
import json, subprocess
import numpy as np
import pyloudnorm as pyln
import re

from pydub.silence import detect_nonsilent
from pydub import AudioSegment, silence

import librosa
import soundfile as sf
import noisereduce as nr
from concurrent.futures import ThreadPoolExecutor, as_completed


import logging
import time

# ─── Timer Start ───────────────────────────────────────────────
start_time = time.perf_counter()

# ─── Configuration ─────────────────────────────────────────────────────
# BASE_DIR = Path(r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production")
BASE_DIR = Path(os.getcwd())

local_language = 'duala'
silence_threshold = 1.5
silence_padding_duration = 3
trailing_silence_duration = 1
repeat_local_audio = 1
flag_pad = True
local_language_title = local_language.title()
test_or_production = "test"
test_or_production = "production"


# ─── Filtering Configuration ──────────────────────────────────────────────
# Set USE_FILTERING to False to process all files as before.
USE_FILTERING = True

# Option 1: Filter by a range of sentence numbers.
# If both sentence and chapter ranges are set, 
# the sentence range will be used.
FILTER_SENTENCE_START = 1622
FILTER_SENTENCE_END   = 1624

FILTER_SENTENCE_START = None
FILTER_SENTENCE_END   = None

# Option 2: Filter by a range of chapter numbers.
# FILTER_CHAPTER_START = 1
# FILTER_CHAPTER_END   = 3
FILTER_CHAPTER_START = None
FILTER_CHAPTER_END   = None 

# Global Silent Segments
trailing_silence = AudioSegment.silent(duration=2000)

# ─── Chapter Ranges ─────────────────────────────────────────────────────
chapter_ranges = [
    (1, 173, "Chap1"), (174, 240, "Chap2"), (241, 258, "Chap3"), (259, 407, "Chap4"),
    (408, 543, "Chap5"), (544, 568, "Chap6"), (569, 703, "Chap7"), (704, 788, "Chap8"),
    (789, 930, "Chap9"), (931, 991, "Chap10"), (992, 1011, "Chap11"), (1012, 1036, "Chap12"),
    (1037, 1074, "Chap13"), (1075, 1104, "Chap14"), (1105, 1125, "Chap15"), (1126, 1152, "Chap16"),
    (1153, 1195, "Chap17"), (1196, 1218, "Chap18"), (1219, 1248, "Chap19"), (1249, 1279, "Chap20"),
    (1280, 1303, "Chap21"), (1304, 1366, "Chap22"), (1367, 1407, "Chap23"), (1408, 1471, "Chap24"),
    (1472, 1500, "Chap25"), (1501, 1569, "Chap26"), (1570, 1650, "Chap27"), (1651, 1717, "Chap28"),
    (1718, 1947, "Chap29"), (1948, 1964, "Chap30"), (1965, 1999, "Chap31"), (2000, 2044, "Chap32")
]

Chapters_begining = [start for start, _, _ in chapter_ranges]
Chapters_ending = [end for _, end, _ in chapter_ranges]
Chapters_begining_extended = sorted(Chapters_begining + [i + 1 for i in Chapters_begining])

# ─── Path Setup ─────────────────────────────────────────────────────────
def get_nth_parent_directory(path: Path, n: int) -> Path:
    for _ in range(n):
        path = path.parent
    return path

local_lang_audio_dir_name = f"{local_language_title}Only" if test_or_production == "production" else f"{local_language_title}OnlyTest"
local_language_dir = (BASE_DIR /"assets"/ "Languages" / f"{local_language_title}Phrasebook" / local_lang_audio_dir_name).resolve()
local_audio_path = local_language_dir
eng_audio_path = BASE_DIR/"assets" / "EnglishOnly"
print("Starting audio directory setup...")


# ── LOGGING CONFIG ──────────────────────────────────────────────────────
assets_dir = BASE_DIR / "assets"
log_dir = assets_dir / "Languages" / f"{local_language_title}Phrasebook"/"Logs"
log_dir.mkdir(parents=True, exist_ok=True)

# Use the script name but place it inside the Phrasebook folder
log_file = log_dir / Path(__file__).with_suffix(".log").name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(log_file, mode="w", encoding="utf-8"),
        logging.StreamHandler()
    ]
)


# ─── Subdirectory Check ─────────────────────────────────────────────────
def check_subdirectories(directory: Path) -> list:
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if subdirs:
        print("Subdirectories found:", [str(d.name) for d in subdirs])
    else:
        print("No subdirectories found.")
    return subdirs

# ─── Audio Extraction (legacy; used only when not filtering) ────────────
def extract_audios_and_move_original(base_path: Path, ext: str = ".mp3"):
    audio_files = []
    original_dir = base_path / "original_audios"
    os.makedirs(original_dir, exist_ok=True)

    subdirs_to_move = []

    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)
        if root_path == original_dir or original_dir in root_path.parents:
            continue

        has_mp3 = False
        for f in files:
            if f.endswith(ext):
                has_mp3 = True
                src = root_path / f
                dest = base_path / f
                try:
                    if src.resolve() != dest.resolve():
                        shutil.copy(src, dest)
                        audio_files.append(dest)
                    else:
                        print(f"⚠️ Skipped copying '{src.name}' (source and destination are the same).")
                except Exception as e:
                    print(f"❌ Error copying {src.name}: {e}")

        if not has_mp3:
            print(f"⚠️ Skipped '{root_path.name}': No *{ext} files found.")

        if root_path != base_path and root_path.parent == base_path:
            subdirs_to_move.append(root_path)

    for subdir in subdirs_to_move:
        if subdir.name != "original_audios":
            try:
                shutil.move(str(subdir), original_dir / subdir.name)
            except Exception as e:
                print(f"❌ Error moving folder '{subdir}': {e}")

    print(f"\n✅ Extracted {len(audio_files)} audio files to {base_path}")
    print(f"📁 Moved {len(subdirs_to_move)} subdirectories to '{original_dir.name}'")
    return natsorted(audio_files)

# ─── Silence Detection and Splitting ────────────────────────────────────
def split_audio_on_silence(audio_path: Path, min_silence_len=1500, silence_thresh=-40):
    audio = AudioSegment.from_file(audio_path)
    chunks = silence.split_on_silence(audio,
                                      min_silence_len=min_silence_len,
                                      silence_thresh=silence_thresh,
                                      keep_silence=0)
    return chunks

# ─── File Renaming ──────────────────────────────────────────────────────
def files_rename(directory, prefix="", suffix="", replace="", by="",
                 remove_first=0, remove_last=0,
                 lower_all=False, upper_all=True,
                 extensions=(".mp3", ".wav")):
    if isinstance(extensions, str):
        extensions = tuple(ext.strip() for ext in extensions.split(','))
    if lower_all and upper_all:
        print("Both lower_all and upper_all are set to True. Defaulting to lowercase.")
        upper_all = False
    for filename in os.listdir(directory):
        if not filename.startswith(local_language):
            original_filename = filename
            if lower_all:
                filename = filename.lower()
            elif upper_all:
                filename = filename.upper()
            try:
                new_name = filename.replace(replace, by)
                new_name = new_name[remove_first:]
                new_name = new_name[:len(new_name) - remove_last]
                new_name = prefix + new_name + suffix
            except:
                new_name = filename
            if extensions is None or (isinstance(extensions, tuple) and original_filename.lower().endswith(extensions)):
                old_file = os.path.join(directory, original_filename)
                new_file = os.path.join(directory, new_name)
                try:
                    os.rename(old_file, new_file)
                    print(f"Renamed '{original_filename}' to '{new_name}'")
                except Exception as e:
                    print(f"❌ FAILED to rename '{original_filename}'. Reason: {e}")
            else:
                print(f"Skipped '{original_filename}': Does not match specified extensions")

def pad_and_join_chunks(seg: AudioSegment, apply_padding: bool = True) -> tuple[AudioSegment, int]:
    def trim_leading_trailing_silence(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
        if not nonsilent_ranges:
            return AudioSegment.silent(duration=0)
        start = nonsilent_ranges[0][0]
        end = nonsilent_ranges[-1][1]
        return audio[start:end]

    chunks = silence.split_on_silence(
        seg,
        min_silence_len=int(silence_threshold * 1000),
        silence_thresh=-50,
        keep_silence=False
    )

    head_silence = AudioSegment.silent(duration=trailing_silence_duration * 1000)
    tail_silence = AudioSegment.silent(duration=trailing_silence_duration * 1000)

    if not apply_padding or len(chunks) == 0:
        cleaned = trim_leading_trailing_silence(seg)
        final_audio = head_silence + cleaned + tail_silence
        return normalize(final_audio), 0

    if len(chunks) > 5:
        cleaned = trim_leading_trailing_silence(seg)
        final_audio = head_silence + cleaned + tail_silence
        return normalize(final_audio), 1

    padding_silence = AudioSegment.silent(duration=int(silence_padding_duration * 1000))
    segments_to_join = [head_silence]

    for i, c in enumerate(chunks):
        cleaned_chunk = trim_leading_trailing_silence(c)
        segments_to_join.append(cleaned_chunk)
        if i < len(chunks) - 1:
            segments_to_join.append(padding_silence)

    segments_to_join.append(tail_silence)

    final_audio = sum(segments_to_join, AudioSegment.silent(duration=0))
    final_audio = final_audio * max(1, repeat_local_audio)
    return normalize(final_audio), len(chunks)

def export_padded_audios(files_to_process: list, out_dir: Path):
    """
    Pads and normalizes audio files, exports only new files.
    Skips existing files using a precomputed set (fast).
    """
    bad = []
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = {f.name for f in out_dir.glob("*.mp3")}
    skipped, processed = 0, 0

    for src in files_to_process:
        try:
            src_path = Path(src)
            output_name = f"{src_path.stem}_padded.mp3"

            if output_name in existing_files:
                skipped += 1
                continue

            seg = AudioSegment.from_file(src_path)
            num = get_digits_from_string(src_path.name)

            if num is None:
                final, _ = normalize(seg), 0
            else:
                final, _ = pad_and_join_chunks(seg, apply_padding=bool(flag_pad))

            final.export(out_dir / output_name, bitrate="192k", format="mp3")
            processed += 1

        except Exception as e:
            bad.append(src_path.name)
            logging.error(f"❌ Error processing {src_path.name}: {e}")
            with open(out_dir / "Bad_audios.txt", "a", encoding="utf-8") as fh:
                fh.write(f"{src_path.name}\n")

    logging.info(f"Step (Pad files): {processed} processed, {skipped} skipped, {len(bad)} failed")

def create_audio_map(file_list):
    audio_map = {}
    for file in file_list:
        filename = file.name
        match = re.search(r'_(\d+)', filename)
        if match:
            audio_number = int(match.group(1))
            audio_map[audio_number] = file
        else:
            print(f"⚠️ Could not extract number from: {filename}")
    return audio_map

def merge_bilingual_audio(eng_audio_dir: Path,
                          local_audio_dir: Path,
                          out_dir: Path,
                          target_ids: set[int] = None):
    """
    Merge bilingual audios, skip already merged files quickly.
    Shows progress on screen for processed files.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    existing_files = {f.name for f in out_dir.glob("*.mp3")}
    skipped, processed, failed = 0, 0, 0

    eng_files, _ = get_audio(eng_audio_dir)
    local_files, _ = get_audio(local_audio_dir)
    eng_map = create_audio_map(eng_files)
    local_map = create_audio_map(local_files)

    total = len(local_map) if not target_ids else len([i for i in local_map if i in target_ids])
    logging.info(f"▶️ Merging bilingual audio for {total} files...")

    for idx, number in enumerate(sorted(local_map), start=1):
        if target_ids and number not in target_ids:
            continue

        local_file = local_map[number]
        eng_file = eng_map.get(number)

        if eng_file:
            output_name = f"{eng_file.stem.split('_')[0]}_{local_file.name}"
        else:
            output_name = f"no_english_{local_file.name}"

        if output_name in existing_files:
            skipped += 1
            continue

        try:
            local_audio = remove_trailing_silence(AudioSegment.from_file(local_file))
            if eng_file:
                eng_audio = remove_trailing_silence(AudioSegment.from_file(eng_file))
                output_audio = Trailing_silent + eng_audio + Inside_silent + local_audio + Trailing_silent
            else:
                output_audio = local_audio

            output_path = out_dir / output_name
            output_audio.export(output_path, format="mp3", bitrate="192k")
            processed += 1

            # ✅ Show progress on screen for processed files
            logging.info(f"[{idx}/{total}] ✅ Created {output_name}")

        except Exception as e:
            failed += 1
            logging.error(f"[{idx}/{total}] ❌ Error processing file #{number}: {e}")

    logging.info(
        f"Step (Merge bilingual): {processed} processed, {skipped} skipped, {failed} failed"
    )

# ─── Helper Functions ───────────────────────────────────────────────────
def get_digits_from_string(s: str):
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None

def normalize(seg: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    change = target_dbfs - seg.dBFS if seg.dBFS != float("-inf") else 0
    return seg.apply_gain(change)

def remove_trailing_silence(audio_segment, silence_thresh=-50, chunk_size=10):
    nonsilent_chunks = detect_nonsilent(audio_segment, min_silence_len=chunk_size, silence_thresh=silence_thresh)
    if nonsilent_chunks:
        start_time = nonsilent_chunks[0][0]
        end_time = nonsilent_chunks[-1][1]
        return audio_segment[start_time:end_time]
    return AudioSegment.silent(duration=0)

def get_audio(directory, ext=["*.mp3", "*.wav"], check_subfolders=False):
    directory = Path(directory)
    files = []
    for e in ext:
        pattern = directory / ("**" if check_subfolders else "") / e
        files += glob.glob(str(pattern), recursive=check_subfolders)
    files = natsorted([Path(f) for f in files], key=lambda x: str(x).lower())
    return files, [f.name for f in files]

def get_chapter(chapter_ranges, page_number):
    for start, end, chapter in chapter_ranges:
        if start <= page_number <= end:
            return chapter
    return None

def get_digits_numbers_from_string(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None

def apply_processing_filter(file_list: list, chapter_ranges: list) -> list:
    """Filters the file list based on global filter configurations."""
    if not USE_FILTERING:
        print("ℹ️  Filtering is disabled. Processing all files.")
        return file_list

    # Default to None
    start_num, end_num = None, None

    # Case 1: Sentence range explicitly defined
    if FILTER_SENTENCE_START is not None and FILTER_SENTENCE_END is not None:
        start_num = FILTER_SENTENCE_START
        end_num = FILTER_SENTENCE_END
        print(f"ℹ️  Filtering by sentence range: {start_num} to {end_num}")

    # Case 2: Chapter range defined (only if sentence range not set)
    elif FILTER_CHAPTER_START is not None and FILTER_CHAPTER_END is not None:
        try:
            start_num = chapter_ranges[FILTER_CHAPTER_START - 1][0]
            end_num = chapter_ranges[FILTER_CHAPTER_END - 1][1]
            print(f"ℹ️  Filtering by chapter range: {FILTER_CHAPTER_START} to {FILTER_CHAPTER_END} "
                  f"(sentences {start_num} to {end_num})")
        except IndexError:
            print(f"❌ Error: Invalid chapter range specified.")
            return []

    # No valid filter → return everything
    if start_num is None:
        print("⚠️  Filtering is enabled, but no valid range is defined. Processing all files.")
        return file_list

    # Apply filter
    filtered_files = []
    for file_path in file_list:
        file_name = Path(file_path).name
        num = get_digits_numbers_from_string(file_name)
        if num and start_num <= num <= end_num:
            filtered_files.append(file_path)

    print(f"✅ Filter applied. Found {len(filtered_files)} files to process.")
    return filtered_files

def file_duration_sec(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)

# ─── Adaptive Threshold ───────────────────────────────────────────────
def adaptive_noise_threshold(path: str, offset_db: float = -25.0) -> float:
    """
    Computes silence threshold relative to integrated loudness.
    Example: offset_db = -25 means '25 dB below average loudness'.
    """
    try:
        data, rate = sf.read(path, always_2d=False)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        I = pyln.Meter(rate).integrated_loudness(data)
        return I + offset_db
    except Exception:
        return -40.0  # fallback conservative threshold

# ─── Silence Detection ────────────────────────────────────────────────
def detect_silences(path: str,
                    min_dur: float = 0.4,
                    offset_db: float = -25.0,
                    min_gap: float = 0.8,
                    fixed_threshold: float = None):
    """
    Detects silence regions.
    - If fixed_threshold is given, use it directly.
    - Otherwise, compute adaptive threshold with offset_db.
    Keeps only gaps longer than min_gap to avoid chopping quiet speech.
    """
    if fixed_threshold is not None:
        noise_db = fixed_threshold
    else:
        noise_db = adaptive_noise_threshold(path, offset_db=offset_db)

    cmd = [
        "ffmpeg","-hide_banner","-nostats","-i", path,
        "-af", f"silencedetect=noise={noise_db:.1f}dB:d={min_dur}",
        "-f","null","-"
    ]
    run = subprocess.run(cmd, text=True, capture_output=True)

    silences = []
    cur_start = None
    for line in run.stderr.splitlines():
        line = line.strip()
        if "silence_start:" in line:
            try:
                cur_start = float(line.split("silence_start:")[1].strip())
            except Exception:
                cur_start = None
        elif "silence_end:" in line and cur_start is not None:
            try:
                seg = line.split("silence_end:")[1].strip()
                end = float(seg.split("|")[0].strip())
                if (end - cur_start) >= min_gap:
                    silences.append((cur_start, end))
            except Exception:
                pass
            cur_start = None

    if cur_start is not None:
        silences.append((cur_start, file_duration_sec(path)))

    return silences

# ─── Batch Processing (with fixed or adaptive silence) ─────────────────

def batch_process_mp3s(
    input_folder: str,
    output_folder: str,
    *,
    files_to_process: list = None,
    silence_noise_db: float = None,  # fixed threshold if provided
    silence_min_dur: float = 0.4,
    offset_db: float = -25.0,
    edge_guard_ms: int = 80,
    apply_silence: bool = True,
    soft_mute: bool = True,
    preset: str = "medium"
) -> None:
    """
    Normalize and process MP3 files.
    Skips already normalized files quickly using a precomputed set.
    """

    TARGET_SR = 44100       # 44.1 kHz
    TARGET_CH = 1           # Mono
    TARGET_BR = "192k"      # MP3 bitrate

    in_dir, out_dir = Path(input_folder), Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(f) for f in files_to_process] if files_to_process else natsorted(in_dir.glob("*.mp3"), key=lambda p: p.name)

    # Precompute existing outputs
    existing_files = {f.name for f in out_dir.glob("*.mp3")}
    skipped, processed, failed = 0, 0, 0
    guard = edge_guard_ms / 1000.0

    for src in files:
        output_name = src.name
        output_path = out_dir / output_name

        if output_name in existing_files:
            skipped += 1
            continue

        safe_silences = []
        if apply_silence:
            silences = detect_silences(
                str(src),
                min_dur=silence_min_dur,
                offset_db=offset_db,
                fixed_threshold=silence_noise_db
            )
            for s, e in silences:
                s2, e2 = s + guard, e - guard
                if e2 > s2:
                    safe_silences.append((s2, e2))

        # Filter chain presets
        if preset == "light":
            af_parts = [
                "highpass=f=80", "lowpass=f=9000",
                "dynaudnorm=f=150:g=10:n=1:p=0.7:m=5",
                "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary",
                "alimiter=limit=-1.5dB:level=true"
            ]
        elif preset == "medium":
            af_parts = [
                "highpass=f=80", "lowpass=f=9000",
                "afftdn=nr=6:nt=w:om=o",
                "dynaudnorm=f=150:g=10:n=1:p=0.7:m=5",
                "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary",
                "alimiter=limit=-1.5dB:level=true"
            ]
        elif preset == "aggressive":
            af_parts = [
                "highpass=f=80", "lowpass=f=9000",
                "afftdn=nr=12:nt=w:om=o",
                "acompressor=threshold=-22dB:ratio=3:attack=8:release=120:knee=3:makeup=3",
                "dynaudnorm=f=150:g=15:n=1:p=0.7:m=7",
                "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=summary",
                "alimiter=limit=-1.5dB:level=true"
            ]
        else:
            raise ValueError(f"Unknown preset: {preset}")

        if apply_silence:
            for (t0, t1) in safe_silences:
                if soft_mute:
                    af_parts.append(f"volume=enable='between(t,{t0:.3f},{t1:.3f})':volume=0.2")
                else:
                    af_parts.append(f"volume=enable='between(t,{t0:.3f},{t1:.3f})':volume=0")

        af_chain = ",".join(af_parts)

        cmd = [
            "ffmpeg", "-y", "-i", str(src),
            "-vn",
            "-af", af_chain,
            "-ar", str(TARGET_SR),
            "-ac", str(TARGET_CH),
            "-b:a", TARGET_BR,
            str(output_path)
        ]

        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            processed += 1
        except subprocess.CalledProcessError as e:
            failed += 1
            logging.error(f"❌ {src.name} failed\n{e.stderr[:300]}")

    logging.info(f"Step (Normalize): {processed} processed, {skipped} skipped, {failed} failed")

def copy_and_rename(src_file: Path) -> Path | None:
    """
    Copy and rename a file into local_audio_path with standardized prefix.
    Returns the new path if successful, else None.
    """
    try:
        num = get_digits_numbers_from_string(src_file.name)
        if num is None:
            return None

        target_name = f"{prefix}{num}.mp3"
        target_path = local_audio_path / target_name

        if not target_path.exists():
            shutil.copy(src_file, target_path)
            print(f"✅ Copied {src_file.name} → {target_name}")
        else:
            print(f"⏩ Already exists: {target_name}")

        return target_path
    except Exception as e:
        print(f"❌ Error copying {src_file}: {e}")
        return None

def timed_step(step_name, func, *args, **kwargs):
    """
    Helper to log execution time for each step.
    """
    logging.info(f"▶️ Starting {step_name}...")
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    logging.info(f"⏱️ {step_name} completed in {elapsed:.2f} seconds")
    return result


def is_new_file_better(temp_path, existing_path):
    from pydub.utils import mediainfo
    new_dur = float(mediainfo(str(temp_path))["duration"])
    old_dur = float(mediainfo(str(existing_path))["duration"])
    return new_dur > old_dur

#####################################################
# END FUNCTIONS
#####################################################

# ─── Global Timer ─────────────────────────────────────────────
global_start = time.perf_counter()

# ──────────────── SELECT WORKING SET ───────────────
prefix = f"{local_language.lower()}_phrasebook_"
subset_files_now = []

if USE_FILTERING:
    # 1) Ensure raw subdirectories are moved into original_audios
    original_dir = local_audio_path / "original_audios"
    os.makedirs(original_dir, exist_ok=True)

    generated_folders = {
        "original_audios",
        "gen1_normalized",
        "gen2_normalized_padded",
        "gen3_bilingual_sentences",
        "bilingual_sentences_chapters",
    }

    for item in local_audio_path.iterdir():
        print(item)
        if item.is_dir() and item.name not in generated_folders:
            try:
                shutil.move(str(item), original_dir / item.name)
                print(f"📂 Moved {item.name} → {original_dir}")
            except Exception as e:
                print(f"❌ Error moving folder '{item}': {e}")

   
    # 2) Gather all files recursively inside original_audios
    # all_audio_files, _ = get_audio(original_dir, check_subfolders=True)

    all_audio_files, _ = timed_step(
        "Step 3 (Gather Files)",
        get_audio, original_dir, check_subfolders=True
    )
    
    # 3) Apply filter
    # source_files_to_process = apply_processing_filter(all_audio_files, chapter_ranges)
    
    source_files_to_process = timed_step(
        "Step 4 (Apply Filter)",
        apply_processing_filter, all_audio_files, chapter_ranges
    )
     
    if not source_files_to_process:
        print("❌ No files to process after filtering. Exiting.")
        raise SystemExit

    # 4) Parallel copy + rename back to parent
    def parallel_copy():
        subset = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(copy_and_rename, Path(f)) for f in source_files_to_process]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    subset.append(result)
        return subset

    subset_files_now = timed_step("Step 5 (Copy + Rename)", parallel_copy)

    working_input_dir = local_audio_path

else:
    # Legacy mode: extract and rename everything
    subdirs = check_subdirectories(local_audio_path)
    if subdirs:
        try:
            _ = extract_audios_and_move_original(local_audio_path)
        except Exception as e:
            print(f"❌ Error extracting audios: {e}")

    files_rename(local_audio_path,
                 prefix=prefix, suffix="",
                 replace="", by="",
                 remove_first=0, remove_last=0,
                 lower_all=True, upper_all=False,
                 extensions=(".mp3", ".wav"))

    subset_files_now, _ = get_audio(local_audio_path)
    working_input_dir = local_audio_path

# ─────────────── NORMALIZE AUDIO (ONLY THE SUBSET) ──────────────────────
# normalized_audio_path = working_input_dir / "gen1_normalized"
normalized_audio_path = BASE_DIR /"assets" / "Languages" / f"{local_language_title}Phrasebook" / "Results_Audios" / "gen1_normalized"
normalized_audio_path.mkdir(parents=True, exist_ok=True)

# batch_process_mp3s(
#     input_folder=str(working_input_dir),
#     output_folder=str(normalized_audio_path),
#     files_to_process=subset_files_now,   # << only process subset
#     silence_noise_db=-32.0,
#     silence_min_dur=0.22,
#     edge_guard_ms=80,
#     apply_silence=False,
#     preset="medium"
# )

timed_step(
    f"Step 6 (Normalize {len(subset_files_now)} files)",
    batch_process_mp3s,
    input_folder=str(working_input_dir),
    output_folder=str(normalized_audio_path),
    files_to_process=subset_files_now,
    silence_noise_db=-32.0,
    silence_min_dur=0.22,
    edge_guard_ms=80,
    apply_silence=False,
    preset="medium"
)

# ─────────────── PAD NORMALIZED AUDIO ───────────────────────────────────
normalized_files_to_process = [normalized_audio_path / f.name for f in subset_files_now]
# normalized_padded_path = working_input_dir / "gen2_normalized_padded"
normalized_padded_path = BASE_DIR/"assets" / "Languages" / f"{local_language_title}Phrasebook" / "Results_Audios" / "gen2_normalized_padded"
normalized_padded_path.mkdir(parents=True, exist_ok=True)

# export_padded_audios(normalized_files_to_process, out_dir=normalized_padded_path)

timed_step(
    f"Step 7 (Pad {len(normalized_files_to_process)} files)",
    export_padded_audios,
    normalized_files_to_process,
    out_dir=normalized_padded_path
)
# ──────────────  BILINGUAL AUDIO PRODUCTION (SUBSET ONLY) ───────────────
Trailing_silent = AudioSegment.silent(duration=1000)
Inside_silent   = AudioSegment.silent(duration=3000)
# bilingual_output_path = working_input_dir / "gen3_bilingual_sentences"
bilingual_output_path = BASE_DIR /"assets" / "Languages" / f"{local_language_title}Phrasebook" / "Results_Audios" / "gen3_bilingual_sentences"
bilingual_output_path.mkdir(parents=True, exist_ok=True)

# Pass subset IDs directly so only filtered files are processed
target_ids = {get_digits_numbers_from_string(f.name) for f in subset_files_now}

# merge_bilingual_audio(
#     eng_audio_path,
#     normalized_padded_path,
#     bilingual_output_path,
#     target_ids=target_ids
# )

timed_step(
    f"Step 8 (Merge bilingual, {len(target_ids)} IDs)",
    merge_bilingual_audio,
    eng_audio_path,
    normalized_padded_path,
    bilingual_output_path,
    target_ids=target_ids
)

def reduce_noise_from_audio(input_file, output_file):
    """
    Reduces noise from an audio file and saves the result.
    
    Args:
        input_file (str): The path to the input audio file.
        output_file (str): The path where the denoised audio will be saved.
    """
    try:
        # Load the audio file
        # sr=None keeps the original sampling rate of the audio
        y, sr = librosa.load(input_file, sr=None)

        # Apply noise reduction
        # The 'y' is the audio signal, and 'sr' is the sampling rate
        # stationary=True assumes the noise is consistent throughout the audio
        reduced_noise_audio = nr.reduce_noise(y=y, 
                                              sr=sr, 
                                              stationary=True,n_fft=2048,)

        # Save the denoised audio to a new file
        # 'sf.write' handles various audio formats (e.g., .wav, .flac)
        sf.write(output_file, reduced_noise_audio, sr)
        print(f"Noise reduction complete! Denoised audio saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# ──────────────  NOISE REDUCTION (on bilingual output) ───────────────
# This will overwrite existing files instead of creating denoised_ copies
for file in bilingual_output_path.glob("*.mp3"):
    try:
        reduce_noise_from_audio(str(file), str(file))  # overwrite in place
        
        print(f"🔄 Overwritten with denoised: {file.name}")
    except Exception as e:
        print(f"❌ Error denoising {file.name}: {e}")


# ──────────────  PER CHAPTER AUDIO PRODUCTION (SUBSET ONLY) ─────────────
combined_chapters_audio_folder = bilingual_output_path / "bilingual_sentences_chapters"
os.makedirs(combined_chapters_audio_folder, exist_ok=True)

# Collect bilingual files but keep only filtered subset
bilingual_files, _ = get_audio(bilingual_output_path, ext=["*.mp3"], check_subfolders=False)
bilingual_files = [f for f in bilingual_files if get_digits_numbers_from_string(f.name) in target_ids]

# =====================================================
song = None
previous_chapter = None

for audio in bilingual_files:
    base_audio = os.path.basename(audio)
    if "mistake" in str(audio).lower() or base_audio[0].isdigit():
        continue

    audio_num = get_digits_numbers_from_string(base_audio)
    chapter_num = get_chapter(chapter_ranges, audio_num)
    if not chapter_num:
        print(f"⚠️ Could not determine chapter for {base_audio} (num={audio_num})")
        continue

    print(f"→ Processing {base_audio} → {chapter_num}")
    song_i = AudioSegment.from_file(audio)

    if song is None:
        song = song_i
        previous_chapter = chapter_num
        continue

    if chapter_num != previous_chapter:
        output_path = combined_chapters_audio_folder / f"phrasebook_{local_language}_{previous_chapter}.mp3"

            
        if output_path.exists():
            temp_path = output_path.with_suffix(".tmp.mp3")
            song.export(temp_path, format="mp3", bitrate="192k")

            if is_new_file_better(temp_path, output_path):
                shutil.move(str(temp_path), str(output_path))
                print(f"🔄 Overwritten {output_path.name} (new file longer in duration)")
            else:
                temp_path.unlink(missing_ok=True)
                print(f"⏩ Skipped {output_path.name} (existing file longer or equal)")
        else:
            song.export(output_path, format="mp3", bitrate="192k")
            print(f"✅ Exported {output_path.name}")


        song = song_i
        previous_chapter = chapter_num
    else:
        song += song_i

if song and previous_chapter:
    output_path = combined_chapters_audio_folder / f"phrasebook_{local_language}_{previous_chapter}.mp3"
    if output_path.exists():
        temp_path = output_path.with_suffix(".tmp.mp3")
        song.export(temp_path, format="mp3", bitrate="192k")
        new_size = temp_path.stat().st_size
        existing_size = output_path.stat().st_size

        if new_size > existing_size:
            shutil.move(str(temp_path), str(output_path))
            print(f"🔄 Overwritten {output_path.name} (new file larger: {new_size} > {existing_size})")
        else:
            temp_path.unlink(missing_ok=True)
            print(f"⏩ Skipped {output_path.name} (existing file is larger or equal)")
    else:
        song.export(output_path, format="mp3", bitrate="192k")
        print(f"✅ Exported {output_path.name}")

# ─── Timer End ─────────────────────────────────────────────────
end_time = time.perf_counter()
elapsed = end_time - start_time
logging.info(f"Script completed in {elapsed:.2f} seconds.")
