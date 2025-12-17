#!/usr/bin/env python3
"""Simplified audio merge pipeline (variant of step1).

Behavior:
- Lecture mode: for each sentence id N produce: EnglishAudioN + pause + N.mp3 + pause + N_.mp3
- Homework mode: for each sentence id N produce: N_.mp3 only

Defaults read `USE_PRIVATE_ASSETS` env and `step0_config` for base dirs.
"""
from __future__ import annotations

import os
import shutil
import logging
import sys
from pathlib import Path
import re
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pydub import AudioSegment, silence
from pydub.silence import detect_nonsilent
from collections import defaultdict
import step0_config as cfg


def _configure_stdio_utf8() -> None:
    # Avoid UnicodeEncodeError on Windows consoles when scripts print emojis/symbols.
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is None:
            continue
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()


USE_PRIVATE_ASSETS = os.getenv(
    "USE_PRIVATE_ASSETS",
    "1" if getattr(cfg, "USE_PRIVATE_ASSETS", True) else "0",
) == "1"

# Chapter ranges (copied from step1_audio_processing) — used only to compute no-repeat IDs
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

# Build set of sentence IDs that should NOT be repeated: the chapter start and the next one
NO_REPEAT_IDS = set()
for start, end, _ in chapter_ranges:
    NO_REPEAT_IDS.add(start)
    NO_REPEAT_IDS.add(start + 1)

def _get_max_workers() -> int:
    """
    Worker count for parallel audio work.
    Priority:
      1) env AUDIO_MAX_WORKERS / MAX_WORKERS
      2) cfg.AUDIO_MAX_WORKERS / cfg.MAX_WORKERS
      3) cpu_count() // 2
    """
    for env_key in ("AUDIO_MAX_WORKERS", "MAX_WORKERS"):
        v = os.getenv(env_key)
        if v:
            try:
                return max(1, int(v))
            except Exception:
                pass

    for cfg_key in ("AUDIO_MAX_WORKERS", "MAX_WORKERS"):
        v = getattr(cfg, cfg_key, None)
        if v is not None:
            try:
                return max(1, int(v))
            except Exception:
                pass

    cpu = os.cpu_count() or 1
    return max(1, cpu // 2)


def _try_ffmpeg_concat_mp3(input_files: list[Path], output_path: Path, *, reencode_bitrate: str = "192k") -> bool:
    """
    Concatenate MP3 files with ffmpeg.

    Tries stream copy first (-c copy). If that fails (mismatched params, etc),
    falls back to a single re-encode.
    """
    if not input_files:
        return False

    ffmpeg_bin = (
        shutil.which("ffmpeg")
        or getattr(cfg, "FFMPEG_BINARY", None)
        or getattr(AudioSegment, "converter", None)
        or "ffmpeg"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
            list_file = f.name
            for p in input_files:
                # concat demuxer prefers forward slashes and needs quoting for spaces
                path_text = str(p.resolve().as_posix()).replace("'", "'\\''")
                f.write(f"file '{path_text}'\n")

        copy_cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c",
            "copy",
            str(output_path),
        ]
        copy_res = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if copy_res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True

        # Retry with re-encode (still faster / lower memory than pydub for large chapters)
        reencode_cmd = [
            ffmpeg_bin,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            list_file,
            "-c:a",
            "libmp3lame",
            "-b:a",
            str(reencode_bitrate),
            str(output_path),
        ]
        reencode_res = subprocess.run(reencode_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return reencode_res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0
    except Exception:
        return False
    finally:
        if list_file:
            try:
                os.remove(list_file)
            except Exception:
                pass


def get_asset_path(relative_path: str) -> Path:
    base = cfg.BASE_DIR / ("private_assets" if USE_PRIVATE_ASSETS else "assets")
    return (base / relative_path).resolve()


def get_digits_numbers_from_string(s: str):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def get_audio(directory: Path, patterns=("*.mp3", "*.wav")):
    files = []
    for pat in patterns:
        files += list(directory.rglob(pat))
    files = sorted(files, key=lambda p: p.name.lower())
    return files


def normalize(seg: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    change = target_dbfs - seg.dBFS if seg.dBFS != float("-inf") else 0
    return seg.apply_gain(change)


def pad_and_join_chunks(seg: AudioSegment, apply_padding: bool = True, repeat_override: int | None = None) -> tuple[AudioSegment, int]:
    SILENCE_THRESH = float(getattr(cfg, "SILENCE_THRESH", 1.5))
    INNER_PAUSE_DURATION = float(getattr(cfg, "INNER_PAUSE_DURATION", 3))
    TRAILING_PAUSE_DURATION = float(getattr(cfg, "TRAILING_PAUSE_DURATION", 1))
    REPEAT_LOCAL_AUDIO = int(getattr(cfg, "REPEAT_LOCAL_AUDIO", 1))
    if isinstance(repeat_override, int) and repeat_override >= 1:
        REPEAT_LOCAL_AUDIO = repeat_override

    def trim_leading_trailing(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
        ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
        if not ranges:
            return AudioSegment.silent(duration=0)
        s, e = ranges[0][0], ranges[-1][1]
        return audio[s:e]

    # New behavior: do NOT split internal audio into chunks for gen2 padding.
    # Always trim leading/trailing silence and then optionally duplicate the whole trimmed audio
    # with a single INNER_PAUSE between repeats. Finally append a trailing pause.
    cleaned = trim_leading_trailing(seg)
    base_audio = cleaned

    inner = AudioSegment.silent(duration=int(INNER_PAUSE_DURATION * 1000))
    trailing = AudioSegment.silent(duration=int(TRAILING_PAUSE_DURATION * 1000))

    if not apply_padding:
        final_audio = base_audio + trailing
        return normalize(final_audio), 1

    if REPEAT_LOCAL_AUDIO > 1:
        combined = base_audio + inner + base_audio
        final_audio = combined + trailing
    else:
        final_audio = base_audio + trailing

    return normalize(final_audio), 1


def copy_and_normalize_to_gen1_parallel(src_files: list[Path], out_dir: Path) -> tuple[list[Path], dict[Path,int]]:
    """
    Build gen1 normalized audio files in parallel.

    Returns:
      - produced list of output Paths
      - repeat_map mapping output Path -> repeat count to apply in gen2 padding stage
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[Path]] = {}
    for src in src_files:
        m = re.search(r"(\d+)(.*)$", src.stem)
        if not m:
            continue
        num = int(m.group(1))
        grouped.setdefault(num, []).append(src)

    max_workers = _get_max_workers()
    produced: list[Path] = []
    repeat_map: dict[Path, int] = {}

    def trim_leading_trailing(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
        ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
        if not ranges:
            return AudioSegment.silent(duration=0)
        s, e = ranges[0][0], ranges[-1][1]
        return audio[s:e]

    def process_num(item: tuple[int, list[Path]]) -> tuple[int, Path, int] | None:
        num, files = item
        # pick primary: exact match 'N' if present, else first non-underscore, else any
        primary = None
        underscore = None
        for f in files:
            stem = f.stem
            if stem == str(num):
                primary = f
            if stem.endswith("_"):
                underscore = f
        if primary is None:
            for f in files:
                if not f.stem.endswith("_"):
                    primary = f
                    break
        if primary is None:
            primary = files[0]

        # If running in homework mode, prefer underscore if present and do NOT combine variants.
        try:
            if getattr(cfg, "MODE", "").lower() == "homework":
                if underscore is not None:
                    primary = underscore
                underscore = None
            else:
                # For lecture, still apply chapter-start no-repeat override
                if int(num) in NO_REPEAT_IDS:
                    if underscore is not None:
                        primary = underscore
                    underscore = None
        except Exception:
            pass

        head = AudioSegment.silent(duration=int(getattr(cfg, "TRAILING_PAUSE_DURATION", 1) * 1000))
        trailing = AudioSegment.silent(duration=int(getattr(cfg, "TRAILING_PAUSE_DURATION", 1) * 1000))

        if underscore is None:
            a1 = AudioSegment.from_file(primary)
            trimmed_a1 = trim_leading_trailing(a1)
            combined = normalize(head + trimmed_a1 + trailing)

            # Repeat behavior for sentence IDs with only one local variant (no "N_.mp3").
            # Default is 2 in lecture mode (so learners hear the same line twice),
            # but you can disable duplication by setting cfg.REPEAT_LOCAL_AUDIO = 1.
            try:
                repeat = max(1, int(getattr(cfg, "REPEAT_LOCAL_AUDIO", 2)))
            except Exception:
                repeat = 2
            try:
                if getattr(cfg, "MODE", "").lower() == "homework":
                    repeat = 1
                else:
                    if int(num) in NO_REPEAT_IDS:
                        repeat = 1
            except Exception:
                pass
        else:
            a1 = AudioSegment.from_file(primary)
            a2 = AudioSegment.from_file(underscore)
            trimmed_a1 = trim_leading_trailing(a1)
            trimmed_a2 = trim_leading_trailing(a2)
            inner = AudioSegment.silent(duration=int(getattr(cfg, "INNER_PAUSE_DURATION", 3) * 1000))
            combined = normalize(head + trimmed_a1 + inner + trimmed_a2 + trailing)
            repeat = 1

        target_name = f"{cfg.LANGUAGE.lower()}_phrasebook_{num}.mp3"
        target_path = out_dir / target_name
        combined.export(target_path, format="mp3", bitrate="192k")
        return (num, target_path, int(repeat))

    items = sorted(grouped.items(), key=lambda kv: kv[0])
    logging.info(f"Building gen1 in parallel: {len(items)} sentence IDs, workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_num, item): item[0] for item in items}
        for fut in as_completed(futures):
            num = futures[fut]
            try:
                res = fut.result()
                if res is None:
                    continue
                _, target_path, repeat = res
                produced.append(target_path)
                repeat_map[target_path] = int(repeat)
            except Exception as e:
                logging.error(f"ƒ?O Copy/normalize {num}: {e}")

    produced.sort(key=lambda p: get_digits_numbers_from_string(p.name) or 0)
    return produced, repeat_map


def export_padded_audios(files_to_process: list, out_dir: Path, repeat_map: dict[Path,int] | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in out_dir.glob("*.mp3")}
    max_workers = _get_max_workers()
    for src in files_to_process:
        try:
            src_path = Path(src)
            # preserve the full stem when creating padded filenames so variants remain distinct
            stem = src_path.stem
            output_name = f"{stem}_padded.mp3"
            if output_name in existing:
                continue
            # Determine repeat behavior: 1 => single copy, 2 => duplicate whole gen1 file with inner pause.
            repeat = 1
            if repeat_map and src_path in repeat_map:
                try:
                    repeat = int(repeat_map[src_path])
                except Exception:
                    repeat = 1

            dest = out_dir / output_name
            if repeat <= 1:
                # Fast path: gen2 should match gen1 exactly when already combined, so do a binary copy.
                try:
                    shutil.copy2(src_path, dest)
                    continue
                except Exception:
                    try:
                        with open(src_path, "rb") as r, open(dest, "wb") as w:
                            shutil.copyfileobj(r, w)
                        continue
                    except Exception:
                        # fall back to decode/encode below
                        logging.debug(f"fallback: failed binary copy for {src_path}, will process instead")
            # Trim leading/trailing silence from the gen1 audio before adding configured padding
            def trim_leading_trailing(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
                ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
                if not ranges:
                    return AudioSegment.silent(duration=0)
                s, e = ranges[0][0], ranges[-1][1]
                return audio[s:e]

            seg = AudioSegment.from_file(src_path)
            trimmed = trim_leading_trailing(seg)

            # Use configurable durations from cfg, fall back to sensible defaults
            INNER_PAUSE_DURATION = float(getattr(cfg, "INNER_PAUSE_DURATION", 3))
            TRAILING_PAUSE_DURATION = float(getattr(cfg, "TRAILING_PAUSE_DURATION", 1))
            inner = AudioSegment.silent(duration=int(INNER_PAUSE_DURATION * 1000))
            head = AudioSegment.silent(duration=int(TRAILING_PAUSE_DURATION * 1000))
            trailing = AudioSegment.silent(duration=int(TRAILING_PAUSE_DURATION * 1000))

            if repeat <= 1:
                final = head + trimmed + trailing
            elif repeat == 2:
                final = head + trimmed + inner + trimmed + trailing
            else:
                parts = [head]
                for i in range(repeat):
                    parts.append(trimmed)
                    if i < repeat - 1:
                        parts.append(inner)
                parts.append(trailing)
                final = sum(parts, AudioSegment.silent(duration=0))

            final = normalize(final)
            final.export(out_dir / output_name, bitrate="192k", format="mp3")
        except Exception as e:
            logging.error(f"❌ Error padding {src}: {e}")


def export_padded_audios_parallel(files_to_process: list[Path], out_dir: Path, repeat_map: dict[Path, int] | None = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in out_dir.glob("*.mp3")}
    max_workers = _get_max_workers()

    def trim_leading_trailing(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
        ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
        if not ranges:
            return AudioSegment.silent(duration=0)
        s, e = ranges[0][0], ranges[-1][1]
        return audio[s:e]

    inner_secs = float(getattr(cfg, "INNER_PAUSE_DURATION", 3))
    trailing_secs = float(getattr(cfg, "TRAILING_PAUSE_DURATION", 1))
    inner = AudioSegment.silent(duration=int(inner_secs * 1000))
    head = AudioSegment.silent(duration=int(trailing_secs * 1000))
    trailing = AudioSegment.silent(duration=int(trailing_secs * 1000))

    def process_one(src: Path) -> None:
        src_path = Path(src)
        stem = src_path.stem
        output_name = f"{stem}_padded.mp3"
        dest = out_dir / output_name
        if output_name in existing or dest.exists():
            return

        repeat = 1
        if repeat_map and src_path in repeat_map:
            try:
                repeat = int(repeat_map[src_path])
            except Exception:
                repeat = 1

        if repeat <= 1:
            # Fast path: gen2 should match gen1 exactly, so copy bytes.
            try:
                shutil.copy2(src_path, dest)
                return
            except Exception:
                try:
                    with open(src_path, "rb") as r, open(dest, "wb") as w:
                        shutil.copyfileobj(r, w)
                    return
                except Exception:
                    pass

        seg = AudioSegment.from_file(src_path)
        trimmed = trim_leading_trailing(seg)

        if repeat <= 1:
            final = head + trimmed + trailing
        elif repeat == 2:
            final = head + trimmed + inner + trimmed + trailing
        else:
            parts = [head]
            for i in range(repeat):
                parts.append(trimmed)
                if i < repeat - 1:
                    parts.append(inner)
            parts.append(trailing)
            final = sum(parts, AudioSegment.silent(duration=0))

        final = normalize(final)
        final.export(dest, bitrate="192k", format="mp3")

    logging.info(f"Padding gen1 -> gen2 in parallel: {len(files_to_process)} files, workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, Path(p)): p for p in files_to_process}
        for fut in as_completed(futures):
            src = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logging.error(f"Error padding {src}: {e}")


def merge_bilingual_padded(eng_dir: Path, local_padded_dir: Path, out_dir: Path, mode: str = "lecture"):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in out_dir.glob("*.mp3")}
    eng_files = get_audio(eng_dir)
    eng_map = {get_digits_numbers_from_string(f.name): f for f in eng_files if get_digits_numbers_from_string(f.name)}
    local_files = list(local_padded_dir.glob("*_padded.mp3"))

    head = AudioSegment.silent(duration=int(getattr(cfg, 'TRAILING_PAUSE_DURATION', 1) * 1000))
    # Use homework-specific inner pause if in homework mode
    inner_secs = float(getattr(cfg, 'INNER_PAUSE_DURATION_HW', getattr(cfg, 'INNER_PAUSE_DURATION', 3))) if mode == 'homework' else float(getattr(cfg, 'INNER_PAUSE_DURATION', 3))
    mid = AudioSegment.silent(duration=int(inner_secs * 1000))

    for f in sorted(local_files, key=lambda p: p.name):
        num = get_digits_numbers_from_string(f.name)
        if num is None:
            continue
        eng = eng_map.get(num)
        # prefix with 'english_' to create unique paired filename preserving variant suffix
        out_name = f"english_{f.name}"
        if out_name in existing:
            continue
        parts = [head]
        local_seg = remove_trailing_silence(AudioSegment.from_file(f))
        if eng:
            eng_seg = remove_trailing_silence(AudioSegment.from_file(eng))
            if mode == 'lecture':
                parts.append(eng_seg)
                parts.append(mid)
                parts.append(local_seg)
            else:
                # homework: local repeated before English
                parts.append(local_seg)
                parts.append(mid)
                parts.append(local_seg)
                parts.append(mid)
                parts.append(eng_seg)
        else:
            parts.append(local_seg)
        parts.append(head)
        combined = sum(parts, AudioSegment.silent(duration=0))
        (out_dir / out_name).parent.mkdir(parents=True, exist_ok=True)
        combined.export(out_dir / out_name, format="mp3", bitrate="192k")


def merge_bilingual_padded_parallel(eng_dir: Path, local_padded_dir: Path, out_dir: Path, mode: str = "lecture"):
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = {f.name for f in out_dir.glob("*.mp3")}
    eng_files = get_audio(eng_dir)
    eng_map = {get_digits_numbers_from_string(f.name): f for f in eng_files if get_digits_numbers_from_string(f.name)}
    local_files = sorted(list(local_padded_dir.glob("*_padded.mp3")), key=lambda p: p.name)

    head = AudioSegment.silent(duration=int(getattr(cfg, "TRAILING_PAUSE_DURATION", 1) * 1000))
    inner_secs = (
        float(getattr(cfg, "INNER_PAUSE_DURATION_HW", getattr(cfg, "INNER_PAUSE_DURATION", 3)))
        if mode == "homework"
        else float(getattr(cfg, "INNER_PAUSE_DURATION", 3))
    )
    mid = AudioSegment.silent(duration=int(inner_secs * 1000))

    max_workers = _get_max_workers()

    def process_one(f: Path) -> None:
        num = get_digits_numbers_from_string(f.name)
        if num is None:
            return
        eng = eng_map.get(num)
        out_name = f"english_{f.name}"
        dest = out_dir / out_name
        if out_name in existing or dest.exists():
            return

        parts = [head]
        local_seg = remove_trailing_silence(AudioSegment.from_file(f))
        if eng:
            eng_seg = remove_trailing_silence(AudioSegment.from_file(eng))
            if mode == "lecture":
                parts.append(eng_seg)
                parts.append(mid)
                parts.append(local_seg)
            else:
                parts.append(local_seg)
                parts.append(mid)
                parts.append(local_seg)
                parts.append(mid)
                parts.append(eng_seg)
        else:
            parts.append(local_seg)
        parts.append(head)

        combined = sum(parts, AudioSegment.silent(duration=0))
        dest.parent.mkdir(parents=True, exist_ok=True)
        combined.export(dest, format="mp3", bitrate="192k")

    logging.info(f"Merging gen2 -> gen3 in parallel: {len(local_files)} files, workers={max_workers}")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_one, f): f for f in local_files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logging.error(f"Error bilingual merge {f}: {e}")


# Chapter concatenation removed to decouple from step1_audio_processing.
# If needed, reintroduce a simple chapter mapping here.


def remove_trailing_silence(audio_segment: AudioSegment, silence_thresh=-50, chunk_size=10):
    nonsilent_chunks = detect_nonsilent(audio_segment, min_silence_len=chunk_size, silence_thresh=silence_thresh)
    if nonsilent_chunks:
        start_time = nonsilent_chunks[0][0]
        end_time   = nonsilent_chunks[-1][1]
        return audio_segment[start_time:end_time]
    return AudioSegment.silent(duration=0)


def get_chapter(chapter_ranges, page_number):
    for start, end, chapter in chapter_ranges:
        if start <= page_number <= end:
            return chapter
    return None


def concatenate_chapters(base_results: Path, bilingual_output_path: Path, chapter_ranges, target_ids: set[int]):
    """Concatenate bilingual files into per-chapter MP3s (writes to Results_Audios/bilingual_sentences_chapters)."""
    # Use a homework-specific folder name so homework outputs are separate
    if getattr(cfg, 'MODE', '').lower() == 'homework':
        combined_chapters_audio_folder = base_results / "homework_bilingual_sentences_chapters"
    else:
        combined_chapters_audio_folder = base_results / "bilingual_sentences_chapters"
    combined_chapters_audio_folder.mkdir(parents=True, exist_ok=True)

    bilingual_files = list(bilingual_output_path.glob("*.mp3"))
    bilingual_files.sort(key=lambda p: (get_digits_numbers_from_string(p.name) or 0, p.name.lower()))

    chapter_range_by_name = {chap: (start, end) for start, end, chap in chapter_ranges}

    # Restrict by START_CHAPTER/END_CHAPTER if provided in cfg
    start_num, end_num = None, None
    if getattr(cfg, 'START_CHAPTER', None) is not None and getattr(cfg, 'END_CHAPTER', None) is not None:
        try:
            start_num = chapter_ranges[cfg.START_CHAPTER - 1][0]
            end_num = chapter_ranges[cfg.END_CHAPTER - 1][1]
        except Exception:
            logging.warning("Invalid START_CHAPTER/END_CHAPTER; ignoring chapter filter")

    # Filter files that are in target_ids and inside chapter range
    filtered_files = []
    for f in bilingual_files:
        num = get_digits_numbers_from_string(f.name)
        if num is None:
            continue
        if num in target_ids and (start_num is None or (start_num <= num <= end_num)):
            filtered_files.append(f)

    chapter_to_files = defaultdict(list)
    for f in filtered_files:
        num = get_digits_numbers_from_string(f.name)
        chap = get_chapter(chapter_ranges, num)
        if chap:
            chapter_to_files[chap].append(f)

    # Ensure deterministic ordering inside each chapter (especially Chap1 where lexicographic sort breaks numeric order)
    for chap, files in chapter_to_files.items():
        files.sort(key=lambda p: (get_digits_numbers_from_string(p.name) or 0, p.name.lower()))

    rng = None
    try:
        import random
        rng = random.Random(getattr(cfg, 'SHUFFLE_SEED', None))
    except Exception:
        rng = None

    for chap, files in sorted(chapter_to_files.items(), key=lambda x: x[0]):
        # apply homework shuffle if configured
        if getattr(cfg, 'SHUFFLE_HOMEWORK', False) and cfg.MODE == 'homework':
            if rng:
                rng.shuffle(files)

        present_ids = [get_digits_numbers_from_string(p.name) for p in files]
        present_ids = sorted({i for i in present_ids if isinstance(i, int)})
        if not present_ids:
            continue

        expected_start, expected_end = chapter_range_by_name.get(chap, (present_ids[0], present_ids[-1]))
        if start_num is not None and end_num is not None:
            expected_start = max(int(expected_start), int(start_num))
            expected_end = min(int(expected_end), int(end_num))

        missing_count = 0
        missing_at_start: list[int] = []
        try:
            present_set = set(present_ids)
            missing = [i for i in range(int(expected_start), int(expected_end) + 1) if i not in present_set]
            missing_count = len(missing)
            missing_at_start = [i for i in range(int(expected_start), min(int(expected_start) + 3, int(expected_end) + 1)) if i in missing]
        except Exception:
            pass

        shuffle_on = bool(getattr(cfg, 'SHUFFLE_HOMEWORK', False) and cfg.MODE == 'homework')
        if missing_count:
            logging.warning(
                f"{chap}: missing {missing_count} sentence IDs in {expected_start}-{expected_end} (first={present_ids[0]})"
            )
            if missing_at_start:
                logging.warning(f"{chap}: missing at chapter start: {missing_at_start}")

        logging.info(
            f"Concatenating {len(files)} files for {chap} (first={present_ids[0]} last={present_ids[-1]} shuffle={shuffle_on})"
        )
        output_path = combined_chapters_audio_folder / f"phrasebook_{cfg.LANGUAGE}_{chap}.mp3"

        if _try_ffmpeg_concat_mp3(files, output_path, reencode_bitrate="192k"):
            logging.info(f"Exported chapter audio: {output_path}")
            continue

        logging.warning(f"ffmpeg concat failed for {chap}; falling back to pydub (slower).")
        combined = None
        for f in files:
            try:
                seg = AudioSegment.from_file(f)
                combined = seg if combined is None else combined + seg
            except Exception as e:
                logging.error(f"Failed to read {f}: {e}")

        if combined is not None:
            try:
                combined.export(output_path, format="mp3", bitrate="192k")
                logging.info(f"Exported chapter audio: {output_path}")
            except Exception as e:
                logging.error(f"Failed to export chapter {chap}: {e}")


# `merge_variant` removed (mixed_variant outputs are no longer generated).


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    LOCAL_LANG_TITLE = cfg.LANGUAGE.title()
    local_lang_dir = get_asset_path(f"Languages/{LOCAL_LANG_TITLE}Phrasebook/{LOCAL_LANG_TITLE}Only")
    eng_dir = get_asset_path("EnglishOnly")
    base_results = get_asset_path(f"Languages/{LOCAL_LANG_TITLE}Phrasebook/Results_Audios")

    # target folders for 3 levels
    normalized_dir = base_results / f"{cfg.MODE}_gen1_normalized"
    padded_dir = base_results / f"{cfg.MODE}_gen2_normalized_padded"
    bilingual_dir = base_results / f"{cfg.MODE}_gen3_bilingual_sentences"

    mode = cfg.MODE.lower()
    logging.info(f"Running 3-level mix pipeline (mode={mode}) -> local={local_lang_dir} eng={eng_dir} results={base_results}")

    # Step A: find source audio files (original_audios preferred)
    possible_original = local_lang_dir / "original_audios"
    source_dir = possible_original if possible_original.exists() else local_lang_dir

    # Clean previous generated folders for a fresh run
    for d in (
        normalized_dir,
        padded_dir,
        bilingual_dir,
        base_results / "bilingual_sentences_chapters",
        base_results / "homework_bilingual_sentences_chapters",
    ):
        try:
            if d.exists():
                shutil.rmtree(d)
        except Exception:
            pass

    # Ensure the standard result folders exist at Results_Audios/*
    for d in (
        normalized_dir,
        padded_dir,
        bilingual_dir,
        base_results / "bilingual_sentences_chapters",
        base_results / "homework_bilingual_sentences_chapters",
    ):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def copy_and_normalize_to_gen1(src_files: list[Path], out_dir: Path) -> tuple[list[Path], dict[Path,int]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        produced = []

        # Group source files by numeric id
        grouped: dict[int, list[Path]] = {}
        repeat_map: dict[Path,int] = {}
        for src in src_files:
            m = re.search(r"(\d+)(.*)$", src.stem)
            if not m:
                continue
            num = int(m.group(1))
            grouped.setdefault(num, []).append(src)

        head = AudioSegment.silent(duration=int(getattr(cfg, 'TRAILING_PAUSE_DURATION', 1) * 1000))
        mid = AudioSegment.silent(duration=int(getattr(cfg, 'INNER_PAUSE_DURATION', 3) * 1000))

        for num, files in sorted(grouped.items()):
            try:
                # pick primary: exact match 'N' if present, else first non-underscore, else any
                primary = None
                underscore = None
                for f in files:
                    stem = f.stem
                    if stem == str(num):
                        primary = f
                    if stem.endswith("_"):
                        underscore = f
                if primary is None:
                    for f in files:
                        if not f.stem.endswith("_"):
                            primary = f
                            break
                if primary is None:
                    primary = files[0]

                # If running in homework mode, prefer the underscore variant if present
                # (use xx_.mp3) and do NOT combine both variants into gen1.
                try:
                    if getattr(cfg, 'MODE', '').lower() == 'homework':
                        if underscore is not None:
                            primary = underscore
                        underscore = None
                    else:
                        # For lecture, still apply chapter-start no-repeat override
                        if int(num) in NO_REPEAT_IDS:
                            if underscore is not None:
                                primary = underscore
                            underscore = None
                except Exception:
                    pass

                # If underscore variant missing: do NOT duplicate into gen1 here.
                # Instead, keep gen1 as a single normalized primary and request repeat at padding stage.
                # helper: trim leading/trailing silence
                def trim_leading_trailing(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
                    ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
                    if not ranges:
                        return AudioSegment.silent(duration=0)
                    s, e = ranges[0][0], ranges[-1][1]
                    return audio[s:e]

                if underscore is None:
                    # No underscore: gen1 should be the primary only.
                    a1 = AudioSegment.from_file(primary)
                    trimmed_a1 = trim_leading_trailing(a1)
                    head = AudioSegment.silent(duration=int(getattr(cfg, 'TRAILING_PAUSE_DURATION', 1) * 1000))
                    trailing = AudioSegment.silent(duration=int(getattr(cfg, 'TRAILING_PAUSE_DURATION', 1) * 1000))
                    combined = normalize(head + trimmed_a1 + trailing)
                    target_name = f"{cfg.LANGUAGE.lower()}_phrasebook_{num}.mp3"
                    target_path = out_dir / target_name
                    combined.export(target_path, format="mp3", bitrate="192k")
                    produced.append(target_path)
                    # request duplication at padding stage so pacing matches underscore variant
                    # In homework mode we never duplicate: always set repeat=1 and prefer underscore variant earlier.
                    try:
                        if getattr(cfg, 'MODE', '').lower() == 'homework':
                            repeat_map[target_path] = 1
                        else:
                            # BUT: do not duplicate for chapter-start sentences and the next sentence
                            if int(num) in NO_REPEAT_IDS:
                                repeat_map[target_path] = 1
                            else:
                                repeat_map[target_path] = 2
                    except Exception:
                        repeat_map[target_path] = 2
                else:
                    # underscore exists: trim both files, insert INNER pause, and add TRAILING pause.
                    a1 = AudioSegment.from_file(primary)
                    a2 = AudioSegment.from_file(underscore)
                    trimmed_a1 = trim_leading_trailing(a1)
                    trimmed_a2 = trim_leading_trailing(a2)
                    head = AudioSegment.silent(duration=int(getattr(cfg, 'TRAILING_PAUSE_DURATION', 1) * 1000))
                    inner = AudioSegment.silent(duration=int(getattr(cfg, 'INNER_PAUSE_DURATION', 3) * 1000))
                    trailing = AudioSegment.silent(duration=int(getattr(cfg, 'TRAILING_PAUSE_DURATION', 1) * 1000))
                    combined = normalize(head + trimmed_a1 + inner + trimmed_a2 + trailing)
                    target_name = f"{cfg.LANGUAGE.lower()}_phrasebook_{num}.mp3"
                    target_path = out_dir / target_name
                    combined.export(target_path, format="mp3", bitrate="192k")
                    produced.append(target_path)
                    # already contains both variants -> no extra repeat
                    repeat_map[target_path] = 1
            except Exception as e:
                logging.error(f"❌ Copy/normalize {num}: {e}")

        return produced, repeat_map

    # gather source files recursively
    all_src = get_audio(source_dir)
    if not all_src:
        logging.error(f"No source audio files found in {source_dir}")
    else:
        # Optional filtering for quicker test runs
        # Canonical config names: START_SENTENCE/END_SENTENCE (also used by step2_video_production).
        # Back-compat: START_ID/END_ID env/cfg names are still accepted.
        start_id = (
            os.getenv("START_SENTENCE")
            or os.getenv("START_ID")
            or getattr(cfg, "START_SENTENCE", None)
            or getattr(cfg, "START_ID", None)
        )
        end_id = (
            os.getenv("END_SENTENCE")
            or os.getenv("END_ID")
            or getattr(cfg, "END_SENTENCE", None)
            or getattr(cfg, "END_ID", None)
        )
        try:
            start_id = int(start_id) if start_id not in (None, "") else None
        except Exception:
            start_id = None
        try:
            end_id = int(end_id) if end_id not in (None, "") else None
        except Exception:
            end_id = None

        if start_id is not None or end_id is not None:
            lo = start_id if start_id is not None else -1_000_000_000
            hi = end_id if end_id is not None else 1_000_000_000
            before = len(all_src)
            all_src = [p for p in all_src if (n := get_digits_numbers_from_string(p.name)) is not None and lo <= n <= hi]
            logging.info(f"Filtering IDs: {lo}..{hi} ({before} files -> {len(all_src)} files)")

        # Gen1: normalized copies with naming local_phrasebook_N (also returns repeat_map)
        gen1_files, gen1_repeat_map = copy_and_normalize_to_gen1_parallel(all_src, normalized_dir)

        # Gen2: padded normalized files (use per-file repeat override)
        export_padded_audios_parallel(gen1_files, padded_dir, repeat_map=gen1_repeat_map)

        # Gen3: bilingual padded merges (english_local_phrasebook_N_padded)
        merge_bilingual_padded_parallel(eng_dir, padded_dir, bilingual_dir, mode=mode)

        # Compute target IDs present in bilingual outputs
        bilingual_files = list(bilingual_dir.glob("*.mp3"))
        target_ids = {get_digits_numbers_from_string(f.name) for f in bilingual_files if get_digits_numbers_from_string(f.name)}

        # Concatenate per chapter (creates Results_Audios/bilingual_sentences_chapters)
        concatenate_chapters(base_results, bilingual_dir, chapter_ranges, target_ids)

        logging.info("Pipeline complete.")
