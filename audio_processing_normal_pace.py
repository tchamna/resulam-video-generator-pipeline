#!/usr/bin/env python3
from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_nonsilent
from natsort import natsorted

import step0_config as cfg


def _configure_stdio_utf8() -> None:
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


def get_asset_path(relative_path: str) -> Path:
    base = cfg.BASE_DIR / ("private_assets" if USE_PRIVATE_ASSETS else "assets")
    return (base / relative_path).resolve()


chapter_ranges = [
    (1, 173, "Chap1"), (174, 240, "Chap2"), (241, 258, "Chap3"), (259, 407, "Chap4"),
    (408, 543, "Chap5"), (544, 568, "Chap6"), (569, 703, "Chap7"), (704, 788, "Chap8"),
    (789, 930, "Chap9"), (931, 991, "Chap10"), (992, 1011, "Chap11"), (1012, 1036, "Chap12"),
    (1037, 1074, "Chap13"), (1075, 1104, "Chap14"), (1105, 1125, "Chap15"), (1126, 1152, "Chap16"),
    (1153, 1195, "Chap17"), (1196, 1218, "Chap18"), (1219, 1248, "Chap19"), (1249, 1279, "Chap20"),
    (1280, 1303, "Chap21"), (1304, 1366, "Chap22"), (1367, 1407, "Chap23"), (1408, 1471, "Chap24"),
    (1472, 1500, "Chap25"), (1501, 1569, "Chap26"), (1570, 1650, "Chap27"), (1651, 1717, "Chap28"),
    (1718, 1947, "Chap29"), (1948, 1964, "Chap30"), (1965, 1999, "Chap31"), (2000, 2044, "Chap32"),
]

# Do not duplicate local audio for chapter headers + immediate next sentence (keeps pacing consistent with old behavior).
NO_REPEAT_IDS: set[int] = set()
for start, _, _ in chapter_ranges:
    NO_REPEAT_IDS.add(start)
    NO_REPEAT_IDS.add(start + 1)


def _get_max_workers() -> int:
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


def _get_ffmpeg_bin() -> str:
    return shutil.which("ffmpeg") or getattr(cfg, "FFMPEG_BINARY", None) or "ffmpeg"


def _try_ffmpeg_concat_mp3(input_files: list[Path], output_path: Path, *, reencode_bitrate: str = "192k") -> bool:
    if not input_files:
        return False
    ffmpeg_bin = _get_ffmpeg_bin()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_file = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
            list_file = f.name
            for p in input_files:
                path_text = str(p.resolve().as_posix()).replace("'", "'\\''")
                f.write(f"file '{path_text}'\n")

        copy_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", list_file,
            "-c", "copy",
            str(output_path),
        ]
        copy_res = subprocess.run(copy_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if copy_res.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True

        reencode_cmd = [
            ffmpeg_bin, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat", "-safe", "0", "-i", list_file,
            "-c:a", "libmp3lame", "-b:a", str(reencode_bitrate),
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


def get_digits_numbers_from_string(s: str) -> int | None:
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else None


def get_audio(directory: Path, patterns=("*.mp3", "*.wav", "*.m4a", "*.aac")) -> list[Path]:
    files: list[Path] = []
    for pat in patterns:
        files += list(directory.rglob(pat))
    files = sorted(files, key=lambda p: p.name.lower())
    return files


def normalize(seg: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    change = target_dbfs - seg.dBFS if seg.dBFS != float("-inf") else 0
    return seg.apply_gain(change)


def remove_trailing_silence(audio_segment: AudioSegment, silence_thresh=-50, chunk_size=10) -> AudioSegment:
    nonsilent_chunks = detect_nonsilent(audio_segment, min_silence_len=chunk_size, silence_thresh=silence_thresh)
    if nonsilent_chunks:
        start_time = nonsilent_chunks[0][0]
        end_time = nonsilent_chunks[-1][1]
        return audio_segment[start_time:end_time]
    return AudioSegment.silent(duration=0)


def _trim_leading_trailing(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
    ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
    if not ranges:
        return AudioSegment.silent(duration=0)
    s, e = ranges[0][0], ranges[-1][1]
    return audio[s:e]


def _export_normal_rythm_underscore(local_dir: Path, out_dir: Path, *, language: str, overwrite: bool, target_ids: set[int] | None = None) -> None:
    """Export audio files (underscore variants preferred, regular variants as fallback) to normal_rythm.
    
    If *target_ids* is given, only those sentence IDs are exported.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    all_files = get_audio(local_dir)
    
    # Group by sentence ID
    by_id: dict[int, list[Path]] = {}
    for p in all_files:
        try:
            # Extract ID from filename (handle both "699" and "699_" formats)
            stem = p.stem
            if stem.endswith("_"):
                sid = int(stem[:-1])
            else:
                sid = int(stem)
            by_id.setdefault(sid, []).append(p)
        except (ValueError, Exception):
            continue
    
    written = 0
    skipped = 0
    for sid, files_for_id in sorted(by_id.items()):
        if target_ids is not None and sid not in target_ids:
            continue
        underscore = next((f for f in files_for_id if f.stem.endswith("_")), None)

        # Prefer underscore variant if it exists, otherwise use the first available
        src = underscore if underscore else files_for_id[0]
        dest = out_dir / f"{language.lower()}_phrasebook_{sid}.mp3"
        
        if dest.exists() and not overwrite:
            continue
        try:
            seg = AudioSegment.from_file(src)
            trimmed = remove_trailing_silence(seg)
            one_sec = AudioSegment.silent(duration=1000)
            final = normalize(one_sec + trimmed + one_sec)
            final.export(dest, format="mp3", bitrate="192k")
            written += 1
        except Exception as e:
            logging.warning(f"Failed to process {src.name} to {dest.name}: {e}")
    
    logging.info("normal_rythm: copied %s file(s), skipped %s chapter starters (no underscore) -> %s", written, skipped, out_dir)


def copy_and_normalize_to_gen1_parallel(
    src_files: list[Path],
    out_dir: Path,
    *,
    language: str,
    mode: str,
    overwrite: bool,
) -> tuple[list[Path], dict[Path, int]]:
    """
    Gen1: For each sentence ID, pick underscore audio if present; otherwise pick a non-underscore file.
    Normal pace rule:
      - If x.mp3 and x_.mp3 exist, ignore x.mp3 and use x_.mp3.
    Duplication happens later in gen2 padding via repeat_map (repeat=2 in lecture).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped: dict[int, list[Path]] = {}
    for src in src_files:
        m = re.search(r"(\d+)(.*)$", src.stem)
        if not m:
            continue
        num = int(m.group(1))
        grouped.setdefault(num, []).append(src)

    max_workers = min(_get_max_workers(), len(grouped) or 1)
    produced: list[Path] = []
    repeat_map: dict[Path, int] = {}

    # Always use exactly 1 second head and tail silence
    one_sec = AudioSegment.silent(duration=1000)
    
    # Get audio source preference from config
    audio_source = getattr(cfg, "AUDIO_SOURCE", "secondary").lower()
    if audio_source not in ("primary", "secondary", "both"):
        audio_source = "secondary"
        logging.warning(f"⚠ Invalid AUDIO_SOURCE={audio_source}, using 'secondary'")

    def process_num(item: tuple[int, list[Path]]) -> tuple[int, Path, int] | None:
        num, files = item

        underscore = None
        primary = None
        for f in files:
            if f.stem.endswith("_"):
                underscore = f
            elif primary is None:
                primary = f

        # Select audio based on AUDIO_SOURCE config
        chosen = None
        if audio_source == "secondary":
            # Prefer underscore variant; fall back to primary if needed
            if underscore:
                try:
                    AudioSegment.from_file(underscore)  # test if readable
                    chosen = underscore
                except Exception as e:
                    logging.warning(f"⚠ Underscore {underscore.name} is corrupted, trying primary: {e}")
                    if primary:
                        chosen = primary
            if chosen is None and primary:
                try:
                    AudioSegment.from_file(primary)  # test if readable
                    chosen = primary
                except Exception as e:
                    logging.warning(f"⚠ Primary {primary.name} is corrupted: {e}")
        
        elif audio_source == "primary":
            # Use primary only, ignore underscore variants
            if primary:
                try:
                    AudioSegment.from_file(primary)  # test if readable
                    chosen = primary
                except Exception as e:
                    logging.warning(f"⚠ Primary {primary.name} is corrupted: {e}")
        
        elif audio_source == "both":
            # Use both if available, otherwise just primary
            if underscore and primary:
                try:
                    AudioSegment.from_file(underscore)
                    AudioSegment.from_file(primary)
                    # For "both", we'll handle this specially in the audio concatenation
                    chosen = underscore  # marker: underscore + primary mode
                except Exception as e:
                    logging.warning(f"⚠ Error loading both files for {num}: {e}")
                    # Fallback to single file
                    chosen = underscore if underscore else primary
            else:
                chosen = underscore if underscore else primary
                if chosen:
                    try:
                        AudioSegment.from_file(chosen)
                    except Exception as e:
                        logging.warning(f"⚠ File {chosen.name} is corrupted: {e}")
                        chosen = None
        
        if chosen is None:
            logging.warning(f"⚠ No readable audio file for sentence {num}")
            return None

        target_name = f"{language.lower()}_phrasebook_{num}.mp3"
        target_path = out_dir / target_name

        if target_path.exists() and not overwrite and target_path.stat().st_size > 0:
            repeat = 1
            if mode == "lecture":
                repeat = 2
                if num in NO_REPEAT_IDS:
                    repeat = 1
            return (num, target_path, int(repeat))

        a1 = AudioSegment.from_file(chosen)
        trimmed = remove_trailing_silence(a1)
        combined = normalize(one_sec + trimmed + one_sec)
        combined.export(target_path, format="mp3", bitrate="192k")

        # Repeat policy: lecture repeats local audio twice unless NO_REPEAT_IDS; homework no repeat.
        repeat = 1
        if mode == "lecture":
            repeat = 2
            if num in NO_REPEAT_IDS:
                repeat = 1
        return (num, target_path, int(repeat))

    items = sorted(grouped.items(), key=lambda kv: kv[0])
    logging.info("Gen1 normal pace: %s sentence IDs, workers=%s", len(items), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_num, item): item[0] for item in items}
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
                logging.error("Gen1 failed for %s: %s", num, e)

    produced.sort(key=lambda p: get_digits_numbers_from_string(p.name) or 0)
    return produced, repeat_map


def export_padded_audios_parallel(files_to_process: list[Path], out_dir: Path, repeat_map: dict[Path, int], *, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    max_workers = min(_get_max_workers(), len(files_to_process) or 1)

    inner_secs = float(getattr(cfg, "INNER_PAUSE_DURATION", 3))
    inner = AudioSegment.silent(duration=int(inner_secs * 1000))
    # Always use exactly 1 second head and tail silence
    one_sec = AudioSegment.silent(duration=1000)

    def process_one(src: Path) -> None:
        stem = src.stem
        output_name = f"{stem}_padded.mp3"
        dest = out_dir / output_name
        if dest.exists() and not overwrite:
            return

        repeat = int(repeat_map.get(src, 1))

        if repeat <= 1:
            # Fast path: gen2 should match gen1 exactly, so copy bytes.
            try:
                shutil.copy2(src, dest)
                return
            except Exception:
                try:
                    with open(src, "rb") as r, open(dest, "wb") as w:
                        shutil.copyfileobj(r, w)
                    return
                except Exception:
                    pass

        seg = AudioSegment.from_file(src)
        trimmed = remove_trailing_silence(seg)

        if repeat <= 1:
            final = one_sec + trimmed + one_sec
        elif repeat == 2:
            final = one_sec + trimmed + inner + trimmed + one_sec
        else:
            parts = [one_sec]
            for i in range(repeat):
                parts.append(trimmed)
                if i < repeat - 1:
                    parts.append(inner)
            parts.append(one_sec)
            final = sum(parts, AudioSegment.silent(duration=0))

        final = normalize(final)
        final.export(dest, bitrate="192k", format="mp3")

    logging.info("Gen2 padding: %s files, workers=%s", len(files_to_process), max_workers)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_one, p): p for p in files_to_process}
        for fut in as_completed(futures):
            p = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logging.error("Gen2 failed for %s: %s", p.name, e)


def merge_bilingual_padded_parallel(eng_dir: Path, local_padded_dir: Path, out_dir: Path, *, mode: str, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    max_workers = _get_max_workers()

    eng_files = get_audio(eng_dir)
    eng_map = {get_digits_numbers_from_string(f.name): f for f in eng_files if get_digits_numbers_from_string(f.name)}
    local_files = natsorted(list(local_padded_dir.glob("*_padded.mp3")), key=lambda p: p.name)

    # Always use exactly 1 second head and tail silence
    one_sec = AudioSegment.silent(duration=1000)
    mid = AudioSegment.silent(duration=int(float(getattr(cfg, "INNER_PAUSE_DURATION", 3)) * 1000))

    def process_one(f: Path) -> None:
        num = get_digits_numbers_from_string(f.name)
        if num is None:
            return
        eng = eng_map.get(num)
        out_name = f"english_{f.name}"
        dest = out_dir / out_name
        if dest.exists() and not overwrite:
            return

        # Strip ALL silence from inputs so we control head/tail exactly
        local_seg = remove_trailing_silence(AudioSegment.from_file(f))
        if eng:
            eng_seg = remove_trailing_silence(AudioSegment.from_file(eng))
        else:
            # If English is missing, skip the English+mid portion entirely
            logging.warning(f"⚠ English audio missing for sentence {num}, skipping English segment")
            eng_seg = None

        # Build: 1s head + content + 1s tail
        if mode == "lecture":
            if eng_seg is not None:
                combined = one_sec + eng_seg + mid + local_seg + one_sec
            else:
                combined = one_sec + local_seg + one_sec
        else:
            if eng_seg is not None:
                combined = one_sec + local_seg + mid + local_seg + mid + eng_seg + one_sec
            else:
                combined = one_sec + local_seg + mid + local_seg + one_sec
        combined.export(dest, format="mp3", bitrate="192k")

    logging.info("Gen3 bilingual merge: %s files, workers=%s", len(local_files), max_workers)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(local_files) or 1)) as ex:
        futures = {ex.submit(process_one, f): f for f in local_files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logging.error("Gen3 failed for %s: %s", f.name, e)


def get_chapter(page_number: int) -> str | None:
    for start, end, chapter in chapter_ranges:
        if start <= page_number <= end:
            return chapter
    return None


def concatenate_chapters(bilingual_output_dir: Path, out_dir: Path, target_ids: set[int], *, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)

    bilingual_files = natsorted(list(bilingual_output_dir.glob("*.mp3")), key=lambda p: p.name)

    chapter_to_files: dict[str, list[Path]] = defaultdict(list)
    for f in bilingual_files:
        num = get_digits_numbers_from_string(f.name)
        if num is None or num not in target_ids:
            continue
        chap = get_chapter(int(num))
        if chap:
            chapter_to_files[chap].append(f)

    for chap, files in chapter_to_files.items():
        files = natsorted(files, key=lambda p: p.name)
        chapter_to_files[chap] = files

    for chap, files in sorted(chapter_to_files.items(), key=lambda x: x[0]):
        output_path = out_dir / f"phrasebook_{cfg.LANGUAGE}_{chap}.mp3"
        if output_path.exists() and not overwrite and output_path.stat().st_size > 0:
            continue
        logging.info("Concatenating %s files for %s", len(files), chap)
        if _try_ffmpeg_concat_mp3(files, output_path, reencode_bitrate="192k"):
            continue
        combined = None
        for f in files:
            try:
                seg = AudioSegment.from_file(f)
                combined = seg if combined is None else combined + seg
            except Exception as e:
                logging.error("Failed to read %s: %s", f, e)
        if combined is not None:
            combined.export(output_path, format="mp3", bitrate="192k")


def add_background_music_to_chapters(in_dir: Path, out_dir: Path, music_path: Path, *, overwrite: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(in_dir.glob("*.mp3"), key=lambda p: p.name.lower())
    if not files:
        logging.info("No chapter audios found in %s", in_dir)
        return

    music_gain_db = float(getattr(cfg, "MUSIC_GAIN_DB", -20.0))
    music = AudioSegment.from_file(music_path)
    workers = min(_get_max_workers(), len(files))

    def process_one(src: Path) -> None:
        dest = out_dir / src.name
        if dest.exists() and not overwrite:
            return
        speech = AudioSegment.from_file(src)
        music_adj = music + music_gain_db
        if music_adj.duration_seconds <= 0:
            raise RuntimeError("Background music has zero duration.")
        loops = int(speech.duration_seconds // music_adj.duration_seconds) + 1
        music_looped = (music_adj * loops)[: len(speech)]
        mixed = speech.overlay(music_looped)
        mixed.export(dest, format="mp3", bitrate="192k")

    logging.info("Adding background music to %s chapter file(s) -> %s", len(files), out_dir)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(process_one, f): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logging.error("Background mix failed for %s: %s", f.name, e)


def _resolve_music_path(lang_base_dir: Path) -> Path | None:
    lang_name = lang_base_dir.name.replace("Phrasebook", "")
    candidates = [
        f"{lang_name.lower()}_music_background.mp3",
        getattr(cfg, "MUSIC_FILENAME", ""),
    ]
    for name in candidates:
        if not name:
            continue
        p = lang_base_dir / name
        if p.exists():
            return p
        if p.suffix == "":
            alt = p.with_suffix(".mp3")
            if alt.exists():
                return alt
    return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ap = argparse.ArgumentParser(
        description=(
            "Normal-pace audio pipeline that prefers underscore local audios (x_.mp3) when present.\n"
            "If x.mp3 and x_.mp3 exist, x.mp3 is ignored and x_.mp3 is used twice (lecture mode)."
        )
    )
    ap.add_argument(
        "language",
        nargs="?",
        default="",
        help="Language name (e.g. ewondo). Defaults to cfg.LANGUAGE.",
    )
    ap.add_argument(
        "--mode",
        default="",
        help="Override mode: lecture or homework. Defaults to cfg.MODE.",
    )
    args = ap.parse_args()

    language = (args.language or "").strip()
    if not language:
        language = str(getattr(cfg, "LANGUAGE", ""))
    LANGUAGE_TITLE = language.title()
    LANGUAGE_CODE = language.lower()

    mode = (args.mode or "").strip().lower() or str(getattr(cfg, "MODE", "lecture")).lower()
    if mode not in ("lecture", "homework"):
        raise SystemExit(f"Invalid mode: {mode} (expected 'lecture' or 'homework')")

    # Use REBUILD_ALL from config only (single source of truth)
    overwrite = bool(getattr(cfg, "REBUILD_ALL", False))

    lang_base_dir = get_asset_path(f"Languages/{LANGUAGE_TITLE}Phrasebook")
    local_lang_dir = get_asset_path(f"Languages/{LANGUAGE_TITLE}Phrasebook/{LANGUAGE_TITLE}Only")
    eng_dir = get_asset_path("EnglishOnly")

    base_results = get_asset_path(f"Languages/{LANGUAGE_TITLE}Phrasebook/Results_Audios_normal_pace")
    base_results.mkdir(parents=True, exist_ok=True)

    normalized_dir = base_results / f"{mode}_gen1_normalized"
    padded_dir = base_results / f"{mode}_gen2_normalized_padded"
    bilingual_dir = base_results / f"{mode}_gen3_bilingual_sentences"
    normal_rythm_dir = base_results / "normal_rythm"
    bilingual_chapters_dir = base_results / "bilingual_sentences_chapters"
    bilingual_chapters_bg_dir = base_results / "bilingual_sentences_chapters_background"

    logging.info("Normal pace audio pipeline (mode=%s) assets=%s", mode, "private_assets" if USE_PRIVATE_ASSETS else "assets")
    logging.info("Local=%s", local_lang_dir)
    logging.info("English=%s", eng_dir)
    logging.info("Out=%s", base_results)

    # When rebuilding, purge all output directories so stale files
    # (e.g. chapter starters that are now skipped) don't carry over.
    if overwrite:
        for d in [normalized_dir, padded_dir, bilingual_dir, normal_rythm_dir,
                  bilingual_chapters_dir, bilingual_chapters_bg_dir]:
            if d.exists():
                shutil.rmtree(d)
                logging.info("🗑 Cleaned %s", d.name)

    possible_original = local_lang_dir / "original_audios"
    source_dir = possible_original if possible_original.exists() else local_lang_dir

    all_src = get_audio(source_dir)
    if not all_src:
        logging.error("No source audio files found in %s", source_dir)
        raise SystemExit(1)

    # ── Optional sentence-range filtering (mirrors step1_audio_processing_v2) ──
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

    selected_ids = getattr(cfg, "SELECTED_SENTENCE_IDS", None)
    if selected_ids:
        before = len(all_src)
        all_src = [p for p in all_src if (n := get_digits_numbers_from_string(p.name)) is not None and n in selected_ids]
        logging.info("Filtering explicit IDs: %s ids (%s files -> %s files)", len(selected_ids), before, len(all_src))
    elif start_id is not None or end_id is not None:
        lo = start_id if start_id is not None else -1_000_000_000
        hi = end_id if end_id is not None else 1_000_000_000
        before = len(all_src)
        all_src = [p for p in all_src if (n := get_digits_numbers_from_string(p.name)) is not None and lo <= n <= hi]
        logging.info("Filtering IDs: %s..%s (%s files -> %s files)", lo, hi, before, len(all_src))

    # Compute the set of IDs we're actually processing (respects START/END filtering)
    _filtered_ids: set[int] | None = None
    if selected_ids or start_id is not None or end_id is not None:
        _filtered_ids = {n for p in all_src if (n := get_digits_numbers_from_string(p.name)) is not None}

    # Export the underscore-only originals into normal_rythm (optional convenience)
    logging.info(f"📁 Step Normal Rhythm: Creating {normal_rythm_dir.name}...")
    normal_rythm_dir.mkdir(parents=True, exist_ok=True)
    _export_normal_rythm_underscore(source_dir, normal_rythm_dir, language=LANGUAGE_TITLE, overwrite=overwrite, target_ids=_filtered_ids)

    # Gen1: normalize underscore-selected audio (never mixes x + x_)
    logging.info(f"📁 Step Gen1: Creating {normalized_dir.name}...")
    normalized_dir.mkdir(parents=True, exist_ok=True)
    gen1_files, gen1_repeat_map = copy_and_normalize_to_gen1_parallel(
        all_src,
        normalized_dir,
        language=LANGUAGE_TITLE,
        mode=mode,
        overwrite=overwrite,
    )

    # Gen2: padding + (lecture) repeat local audio twice
    logging.info(f"📁 Step Gen2: Creating {padded_dir.name}...")
    padded_dir.mkdir(parents=True, exist_ok=True)
    export_padded_audios_parallel(gen1_files, padded_dir, gen1_repeat_map, overwrite=overwrite)

    # Gen3: bilingual (English + local)
    logging.info(f"📁 Step Gen3: Creating {bilingual_dir.name}...")
    bilingual_dir.mkdir(parents=True, exist_ok=True)
    merge_bilingual_padded_parallel(eng_dir, padded_dir, bilingual_dir, mode=mode, overwrite=overwrite)

    # Concatenate into chapter audios
    bilingual_files = list(bilingual_dir.glob("*.mp3"))
    target_ids = {get_digits_numbers_from_string(f.name) for f in bilingual_files if get_digits_numbers_from_string(f.name)}
    # Ensure output filenames reflect the selected language, not cfg.LANGUAGE.
    # We keep the same internal naming convention as other pipelines.
    logging.info(f"📁 Step Concatenate: Creating {bilingual_chapters_dir.name}...")
    bilingual_chapters_dir.mkdir(parents=True, exist_ok=True)
    old_lang = getattr(cfg, "LANGUAGE", None)
    try:
        cfg.LANGUAGE = LANGUAGE_TITLE  # type: ignore[attr-defined]
        concatenate_chapters(bilingual_dir, bilingual_chapters_dir, target_ids, overwrite=overwrite)
    finally:
        if old_lang is not None:
            try:
                cfg.LANGUAGE = old_lang  # type: ignore[attr-defined]
            except Exception:
                pass

    # Add background music to chapters
    logging.info(f"📁 Step Background Music: Creating {bilingual_chapters_bg_dir.name}...")
    bilingual_chapters_bg_dir.mkdir(parents=True, exist_ok=True)
    music_path = _resolve_music_path(lang_base_dir)
    if music_path:
        add_background_music_to_chapters(bilingual_chapters_dir, bilingual_chapters_bg_dir, music_path, overwrite=overwrite)
    else:
        logging.warning("Background music not found in %s; skipping chapter background.", lang_base_dir)

    logging.info("Done.")
