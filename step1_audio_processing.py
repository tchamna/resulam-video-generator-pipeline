import os
import shutil
import glob
import json
import subprocess
import re
import time
import logging
from pathlib import Path
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyloudnorm as pyln
from natsort import natsorted
from pydub import AudioSegment, silence
from pydub.silence import detect_nonsilent
import librosa
import soundfile as sf
import noisereduce as nr


import random
from collections import defaultdict

# ─── Central config ─────────────────────────────────────────────────────
import step0_config as cfg


MODE = cfg.MODE.lower()                      # "lecture" | "homework"
ENV  = cfg.ENV

# Shuffle controls (default ON for homework)
SHUFFLE_HOMEWORK = bool(getattr(cfg, "SHUFFLE_HOMEWORK", MODE == "homework"))
SHUFFLE_SEED = getattr(cfg, "SHUFFLE_SEED", None)  # None => different each run

local_language = cfg.LANGUAGE.lower()
local_language_title = cfg.LANGUAGE.title()


# ─── USE FILTERING ─────────────────────────────────────────
USE_FILTERING = True

# ─── Timer ──────────────────────────────────────────────────────────────
start_time = time.perf_counter()

# ─── Asset mode (env can override cfg) ──────────────────────────────────
USE_PRIVATE_ASSETS = os.getenv(
    "USE_PRIVATE_ASSETS",
    "1" if cfg.USE_PRIVATE_ASSETS else "0"
) == "1"
print(f"[Config] USE_PRIVATE_ASSETS = {USE_PRIVATE_ASSETS}")

# ─── Bind config values ─────────────────────────────────────────────────

#Get attributes from cfg if it exists; otherwise, use defaults values here
SILENCE_THRESH         = float(getattr(cfg, "SILENCE_THRESH", 1.5))
INNER_PAUSE_DURATION  = float(getattr(cfg, "INNER_PAUSE_DURATION", 3))
TRAILING_PAUSE_DURATION = float(getattr(cfg, "TRAILING_PAUSE_DURATION", 1))
REPEAT_LOCAL_AUDIO        = int(getattr(cfg, "REPEAT_LOCAL_AUDIO", 1))
FLAG_PAD                  = bool(getattr(cfg, "FLAG_PAD", True))

FILTER_SENTENCE_START = getattr(cfg, "FILTER_SENTENCE_START", None)
FILTER_SENTENCE_END   = getattr(cfg, "FILTER_SENTENCE_END", None)
FILTER_CHAPTER_START  = getattr(cfg, "FILTER_CHAPTER_START", None)
FILTER_CHAPTER_END    = getattr(cfg, "FILTER_CHAPTER_END", None)


# ─── Validation ─────────────────────────────────────────────────────────
if MODE not in {"lecture", "homework"}:
    raise ValueError(f"Invalid MODE: {MODE!r}")
if ENV not in {"production", "test"}:
    raise ValueError(f"Invalid ENV: {ENV!r}")

# ─── Chapter ranges ─────────────────────────────────────────────────────
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

# ======================================================================
# Helpers / utilities
# ======================================================================

def get_asset_path(relative_path: str) -> Path:
    """Resolve a path inside either 'assets' or 'private_assets' using cfg.BASE_DIR."""
    base = cfg.BASE_DIR / ("private_assets" if USE_PRIVATE_ASSETS else "assets")
    return base / relative_path

def check_subdirectories(directory: Path) -> list[Path]:
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if subdirs:
        print("Subdirectories:", [d.name for d in subdirs])
    else:
        print("No subdirectories.")
    return subdirs

def extract_audios_and_move_original(base_path: Path, ext: str = ".mp3"):
    audio_files = []
    original_dir = base_path / "original_audios"
    os.makedirs(original_dir, exist_ok=True)
    subdirs_to_move = []

    for root, _, files in os.walk(base_path):
        root_path = Path(root)
        if root_path == original_dir or original_dir in root_path.parents:
            continue

        has_audio = False
        for f in files:
            if f.endswith(ext):
                has_audio = True
                src = root_path / f
                dest = base_path / f
                try:
                    if src.resolve() != dest.resolve():
                        shutil.copy(src, dest)
                        audio_files.append(dest)
                    else:
                        print(f"⚠️ Skipped '{src.name}' (same dest).")
                except Exception as e:
                    print(f"❌ Copy error {src.name}: {e}")

        if not has_audio:
            print(f"⚠️ Skipped '{root_path.name}': No *{ext} files.")

        if root_path != base_path and root_path.parent == base_path:
            subdirs_to_move.append(root_path)

    for subdir in subdirs_to_move:
        if subdir.name != "original_audios":
            try:
                shutil.move(str(subdir), original_dir / subdir.name)
            except Exception as e:
                print(f"❌ Move error '{subdir}': {e}")

    print(f"✅ Extracted {len(audio_files)} to {base_path}")
    print(f"📁 Moved {len(subdirs_to_move)} subdirs to '{original_dir.name}'")
    return natsorted(audio_files)

def split_audio_on_silence(audio_path: Path, min_silence_len=1500, silence_thresh=-40):
    audio = AudioSegment.from_file(audio_path)
    chunks = silence.split_on_silence(
        audio, min_silence_len=min_silence_len, silence_thresh=silence_thresh, keep_silence=0
    )
    return chunks

def files_rename(directory, prefix="", suffix="", replace="", by="",
                 remove_first=0, remove_last=0,
                 lower_all=False, upper_all=True,
                 extensions=(".mp3", ".wav")):
    if isinstance(extensions, str):
        extensions = tuple(ext.strip() for ext in extensions.split(','))
    if lower_all and upper_all:
        print("Both lower_all and upper_all True. Defaulting to lowercase.")
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
            except Exception:
                new_name = filename

            if extensions is None or (isinstance(extensions, tuple) and original_filename.lower().endswith(extensions)):
                old_file = os.path.join(directory, original_filename)
                new_file = os.path.join(directory, new_name)
                try:
                    os.rename(old_file, new_file)
                    print(f"Renamed '{original_filename}' → '{new_name}'")
                except Exception as e:
                    print(f"❌ Rename failed '{original_filename}': {e}")
            else:
                print(f"Skipped '{original_filename}': extension mismatch")

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
        end_time   = nonsilent_chunks[-1][1]
        return audio_segment[start_time:end_time]
    return AudioSegment.silent(duration=0)

def pad_and_join_chunks(seg: AudioSegment, apply_padding: bool = True) -> tuple[AudioSegment, int]:
    def trim_leading_trailing(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
        ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
        if not ranges:
            return AudioSegment.silent(duration=0)
        s, e = ranges[0][0], ranges[-1][1]
        return audio[s:e]

    chunks = silence.split_on_silence(
        seg,
        min_silence_len=int(SILENCE_THRESH * 1000),
        silence_thresh=-50,
        keep_silence=False
    )

    head = AudioSegment.silent(duration=TRAILING_PAUSE_DURATION * 1000)
    tail = AudioSegment.silent(duration=TRAILING_PAUSE_DURATION * 1000)

    if not apply_padding or len(chunks) == 0:
        cleaned = trim_leading_trailing(seg)
        final_audio = head + cleaned + tail
        return normalize(final_audio), 0

    if len(chunks) > 5:
        cleaned = trim_leading_trailing(seg)
        final_audio = head + cleaned + tail
        return normalize(final_audio), 1

    pad = AudioSegment.silent(duration=int(INNER_PAUSE_DURATION * 1000))
    parts = [head]
    for i, c in enumerate(chunks):
        cleaned_chunk = trim_leading_trailing(c)
        parts.append(cleaned_chunk)
        if i < len(chunks) - 1:
            parts.append(pad)
    parts.append(tail)

    final_audio = sum(parts, AudioSegment.silent(duration=0))
    final_audio = final_audio * max(1, REPEAT_LOCAL_AUDIO)
    return normalize(final_audio), len(chunks)

def export_padded_audios(files_to_process: list, out_dir: Path):
    bad = []
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in out_dir.glob("*.mp3")}
    skipped, processed = 0, 0

    for src in files_to_process:
        try:
            src_path = Path(src)
            output_name = f"{src_path.stem}_padded.mp3"
            if output_name in existing:
                skipped += 1
                continue

            seg = AudioSegment.from_file(src_path)
            num = get_digits_from_string(src_path.name)

            if num is None:
                final, _ = normalize(seg), 0
            else:
                final, _ = pad_and_join_chunks(seg, apply_padding=bool(FLAG_PAD))

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
        match = re.search(r'_(\d+)', file.name)
        if match:
            audio_map[int(match.group(1))] = file
        else:
            print(f"⚠️ Could not extract number from: {file.name}")
    return audio_map

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
    if not USE_FILTERING:
        print("ℹ️  Filtering disabled. Processing all files.")
        return file_list

    start_num, end_num = None, None

    if FILTER_SENTENCE_START is not None and FILTER_SENTENCE_END is not None:
        start_num = FILTER_SENTENCE_START
        end_num   = FILTER_SENTENCE_END
        print(f"ℹ️  Sentence range: {start_num}–{end_num}")

    elif FILTER_CHAPTER_START is not None and FILTER_CHAPTER_END is not None:
        try:
            start_num = chapter_ranges[FILTER_CHAPTER_START - 1][0]
            end_num   = chapter_ranges[FILTER_CHAPTER_END - 1][1]
            print(f"ℹ️  Chapter range: {FILTER_CHAPTER_START}–{FILTER_CHAPTER_END} "
                  f"(sentences {start_num}–{end_num})")
        except IndexError:
            print("❌ Invalid chapter range.")
            return []

    if start_num is None:
        print("⚠️  Filtering enabled, but no valid range. Processing all files.")
        return file_list

    filtered = []
    for file_path in file_list:
        num = get_digits_numbers_from_string(Path(file_path).name)
        if num and start_num <= num <= end_num:
            filtered.append(file_path)

    print(f"✅ Filter applied → {len(filtered)} files")
    return filtered

def file_duration_sec(path: str) -> float:
    info = sf.info(path)
    return float(info.frames) / float(info.samplerate)

def adaptive_noise_threshold(path: str, offset_db: float = -25.0) -> float:
    try:
        data, rate = sf.read(path, always_2d=False)
        if not np.issubdtype(data.dtype, np.floating):
            data = data.astype(np.float32) / np.iinfo(data.dtype).max
        I = pyln.Meter(rate).integrated_loudness(data)
        return I + offset_db
    except Exception:
        return -40.0

def detect_silences(path: str,
                    min_dur: float = 0.4,
                    offset_db: float = -25.0,
                    min_gap: float = 0.8,
                    fixed_threshold: float = None):
    noise_db = fixed_threshold if fixed_threshold is not None else adaptive_noise_threshold(path, offset_db=offset_db)
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

def batch_process_mp3s(
    input_folder: str,
    output_folder: str,
    *,
    files_to_process: list = None,
    silence_noise_db: float = None,
    silence_min_dur: float = 0.4,
    offset_db: float = -25.0,
    edge_guard_ms: int = 80,
    apply_silence: bool = True,
    soft_mute: bool = True,
    preset: str = "medium"
) -> None:
    TARGET_SR = 44100
    TARGET_CH = 1
    TARGET_BR = "192k"

    in_dir, out_dir = Path(input_folder), Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = [Path(f) for f in files_to_process] if files_to_process else natsorted(in_dir.glob("*.mp3"), key=lambda p: p.name)

    existing = {f.name for f in out_dir.glob("*.mp3")}
    skipped, processed, failed = 0, 0, 0
    guard = edge_guard_ms / 1000.0

    for src in files:
        output_path = out_dir / src.name
        if src.name in existing:
            skipped += 1
            continue

        safe_silences = []
        if apply_silence:
            sils = detect_silences(
                str(src),
                min_dur=silence_min_dur,
                offset_db=offset_db,
                fixed_threshold=silence_noise_db
            )
            for s, e in sils:
                s2, e2 = s + guard, e - guard
                if e2 > s2:
                    safe_silences.append((s2, e2))

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
                af_parts.append(
                    f"volume=enable='between(t,{t0:.3f},{t1:.3f})':volume={0.2 if soft_mute else 0}"
                )

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
    try:
        num = get_digits_numbers_from_string(src_file.name)
        if num is None:
            return None

        target_name = f"{local_language.lower()}_phrasebook_{num}.mp3"
        target_path = local_audio_path / target_name

        if not target_path.exists():
            shutil.copy(src_file, target_path)
            print(f"✅ Copied {src_file.name} → {target_name}")
        else:
            print(f"⏩ Exists: {target_name}")

        return target_path
    except Exception as e:
        print(f"❌ Copy error {src_file}: {e}")
        return None

def is_new_file_better(temp_path, existing_path):
    from pydub.utils import mediainfo
    new_dur = float(mediainfo(str(temp_path))["duration"])
    old_dur = float(mediainfo(str(existing_path))["duration"])
    return new_dur > old_dur

def reduce_noise_from_audio(input_file, output_file):
    try:
        y, sr = librosa.load(input_file, sr=None)
        reduced = nr.reduce_noise(y=y, sr=sr, stationary=True, n_fft=2048)
        sf.write(output_file, reduced, sr)
        print(f"NR complete → {Path(output_file).name}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except Exception as e:
        print(f"NR error: {e}")

def resolve_local_lang_dir_name(lang_title: str, env: str, mode: str) -> str:
    if env == "production" and mode == "lecture":
        return f"{lang_title}Only"
    if env == "test" and mode == "lecture":
        return f"{lang_title}OnlyTest"
    if env == "production" and mode == "homework":
        return f"{lang_title}OnlyHomework"
    return f"{lang_title}OnlyTestHomework"

def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f} sec"
    if seconds < 3600:
        return f"{seconds/60:.2f} min"
    if seconds < 86400:
        return f"{seconds/3600:.2f} hr"
    return f"{seconds/86400:.2f} days"

@contextmanager
def log_time(step_name: str):
    t0 = time.perf_counter()
    logging.info(f"▶️ {step_name}…")
    try:
        yield
    finally:
        logging.info(f"⏱ {step_name} done in {format_elapsed(time.perf_counter()-t0)}")

# ======================================================================
# Paths & logging
# ======================================================================

local_lang_audio_dir_name = resolve_local_lang_dir_name(local_language_title, ENV, MODE)

local_language_dir = get_asset_path(
    f"Languages/{local_language_title}Phrasebook/{local_lang_audio_dir_name}"
).resolve()
local_audio_path = local_language_dir

eng_audio_path = get_asset_path("EnglishOnly")
log_dir = get_asset_path(f"Languages/{local_language_title}Phrasebook/Logs")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / Path(__file__).with_suffix(".log").name

normalized_audio_path = get_asset_path(
    f"Languages/{local_language_title}Phrasebook/Results_Audios/{MODE}_gen1_normalized"
)
normalized_padded_path = get_asset_path(
    f"Languages/{local_language_title}Phrasebook/Results_Audios/{MODE}_gen2_normalized_padded"
)
bilingual_output_path = get_asset_path(
    f"Languages/{local_language_title}Phrasebook/Results_Audios/{MODE}_gen3_bilingual_sentences"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(log_file, mode="w", encoding="utf-8"),
              logging.StreamHandler()]
)

# ======================================================================
# Main pipeline
# ======================================================================

global_start = time.perf_counter()
prefix = f"{local_language.lower()}_phrasebook_"
subset_files_now: list[Path] = []

if USE_FILTERING:
    # original_dir = local_audio_path / "original_audios"
    original_dir = local_audio_path 
    os.makedirs(original_dir, exist_ok=True)

    generated_folders = {
        "original_audios",
        "gen1_normalized",
        "gen2_normalized_padded",
        "gen3_bilingual_sentences",
        "bilingual_sentences_chapters",
    }

    for item in local_audio_path.iterdir():
        if item.is_dir() and item.name not in generated_folders:
            try:
                shutil.move(str(item), original_dir / item.name)
                print(f"📂 Moved {item.name} → original_audios")
            except Exception as e:
                print(f"❌ Move error '{item}': {e}")

    with log_time("Step 3 (Gather Files)"):
        all_audio_files, _ = get_audio(original_dir, check_subfolders=True)

    with log_time("Step 4 (Apply Filter)"):
        source_files_to_process = apply_processing_filter(all_audio_files, chapter_ranges)

    if not source_files_to_process:
        print("❌ No files after filtering.")
        raise SystemExit

    def parallel_copy():
        subset = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(copy_and_rename, Path(f)) for f in source_files_to_process]
            for fut in as_completed(futures):
                res = fut.result()
                if res:
                    subset.append(res)
        return subset

    with log_time("Step 5 (Copy + Rename)"):
        subset_files_now = parallel_copy()

    working_input_dir = local_audio_path
    
    subdirs = check_subdirectories(local_audio_path)
    if subdirs:
        try:
            _ = extract_audios_and_move_original(local_audio_path)
        except Exception as e:
            print(f"❌ Extract error: {e}")
else:
    subdirs = check_subdirectories(local_audio_path)
    if subdirs:
        try:
            _ = extract_audios_and_move_original(local_audio_path)
        except Exception as e:
            print(f"❌ Extract error: {e}")

    files_rename(local_audio_path,
                 prefix=prefix, suffix="",
                 replace="", by="",
                 remove_first=0, remove_last=0,
                 lower_all=True, upper_all=False,
                 extensions=(".mp3", ".wav"))

    subset_files_now, _ = get_audio(local_audio_path)
    working_input_dir = local_audio_path

# Normalize
with log_time(f"Step 6 (Normalize {len(subset_files_now)} files)"):
    batch_process_mp3s(
        input_folder=str(working_input_dir),
        output_folder=str(normalized_audio_path),
        files_to_process=subset_files_now,
        silence_noise_db=-32.0,
        silence_min_dur=0.22,
        edge_guard_ms=80,
        apply_silence=False,
        preset="medium"
    )

# Pad
normalized_files_to_process = [normalized_audio_path / f.name for f in subset_files_now]
with log_time(f"Step 7 (Pad {len(normalized_files_to_process)} files)"):
    export_padded_audios(normalized_files_to_process, out_dir=normalized_padded_path)

# Merge bilingual
Trailing_silent = AudioSegment.silent(duration=cfg.TRAILING_PAUSE_DURATION * 1000)
Inside_silent   = AudioSegment.silent(duration=cfg.INNER_PAUSE_DURATION * 1000)

def merge_bilingual_audio(
    eng_audio_dir: Path,
    local_audio_dir: Path,
    out_dir: Path,
    target_ids: set[int] = None,
) -> None:
    
    out_dir.mkdir(parents=True, exist_ok=True)
    existing_files = {f.name for f in out_dir.glob("*.mp3")}
    skipped, processed, failed = 0, 0, 0

    eng_files, _   = get_audio(eng_audio_dir)
    local_files, _ = get_audio(local_audio_dir)
    eng_map   = create_audio_map(eng_files)
    local_map = create_audio_map(local_files)

    in_scope_ids = sorted(local_map.keys()) if not target_ids else [i for i in local_map.keys() if i in target_ids]

    total = len(in_scope_ids)
    logging.info(f"▶️ Merging bilingual for {total} files…")

    for idx, number in enumerate(in_scope_ids, start=1):
        local_file = local_map[number]
        eng_file   = eng_map.get(number)

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
                if MODE == "lecture":
                    output_audio = Trailing_silent + eng_audio + Inside_silent + local_audio + Trailing_silent
                else:
                    output_audio = Trailing_silent + (local_audio + Inside_silent) * 2 + eng_audio + Trailing_silent
            else:
                output_audio = local_audio

            (out_dir / output_name).parent.mkdir(parents=True, exist_ok=True)
            output_audio.export(out_dir / output_name, format="mp3", bitrate="192k")
            processed += 1
        except Exception as e:
            failed += 1
            logging.error(f"[{idx}/{total}] ❌ Error #{number}: {e}")

    logging.info(f"Step (Merge bilingual): {processed} processed, {skipped} skipped, {failed} failed")

def concatenate_chapters(bilingual_output_path: Path, chapter_ranges, target_ids: set[int]):
    combined_chapters_audio_folder = bilingual_output_path / "bilingual_sentences_chapters"
    os.makedirs(combined_chapters_audio_folder, exist_ok=True)

    bilingual_files, _ = get_audio(bilingual_output_path, ext=["*.mp3"], check_subfolders=False)
    bilingual_files = [f for f in bilingual_files if get_digits_numbers_from_string(f.name) in target_ids]

    chapter_to_files = defaultdict(list)
    for f in bilingual_files:
        num = get_digits_numbers_from_string(f.name)
        chap = get_chapter(chapter_ranges, num)
        if chap:
            chapter_to_files[chap].append(f)

    rng = random.Random(SHUFFLE_SEED)

    for chap, files in sorted(chapter_to_files.items(), key=lambda x: x[0]):
        # if MODE == "homework":
        #     rng.shuffle(files)
        
        if SHUFFLE_HOMEWORK and (MODE == "homework"):
            rng.shuffle(files)


        print(f"🎵 Concatenating {len(files)} files for {chap} (shuffle={MODE=='homework'})")

        combined = None
        for f in files:
            seg = AudioSegment.from_file(f)
            combined = seg if combined is None else combined + seg

        output_path = combined_chapters_audio_folder / f"phrasebook_{local_language}_{chap}.mp3"
        combined.export(output_path, format="mp3", bitrate="192k")
        print(f"✅ Exported {output_path.name}")

# Merge bilingual (sentence-level, no shuffle)
target_ids = {get_digits_numbers_from_string(f.name) for f in subset_files_now}
with log_time(f"Step 8 (Merge bilingual, {len(target_ids)} IDs)"):
    merge_bilingual_audio(
        eng_audio_path,
        normalized_padded_path,
        bilingual_output_path,
        target_ids=target_ids
    )

# Noise reduction (overwrite in place)
for file in bilingual_output_path.glob("*.mp3"):
    try:
        reduce_noise_from_audio(str(file), str(file))
    except Exception as e:
        print(f"❌ NR error {file.name}: {e}")

# Concatenate per chapter (shuffle applied here if homework)
with log_time("Step 9 (Concatenate per chapter)"):
    concatenate_chapters(bilingual_output_path, chapter_ranges, target_ids)

# ─── End timing ─────────────────────────────────────────────────────────
elapsed = time.perf_counter() - start_time
logging.info(f"Script completed in {format_elapsed(elapsed)}")
