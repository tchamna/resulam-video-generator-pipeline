from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydub import AudioSegment

# Ensure repo root is importable when running from scripts/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import step0_config as cfg


def _resolve_input_dir(raw: str) -> Path:
    p = Path(raw)
    if p.exists():
        return p
    # Try relative to repo root
    repo_rel = Path.cwd() / raw
    if repo_rel.exists():
        return repo_rel
    # Try inside assets/private_assets
    base = Path.cwd() / ("private_assets" if getattr(cfg, "USE_PRIVATE_ASSETS", True) else "assets")
    alt = base / raw
    return alt


def _find_lang_base_dir(input_dir: Path) -> Path | None:
    for parent in [input_dir] + list(input_dir.parents):
        if parent.name.endswith("Phrasebook"):
            return parent
    return None


def _resolve_music_path(input_dir: Path, explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        rel = Path.cwd() / explicit
        if rel.exists():
            return rel
        raise FileNotFoundError(f"Music file not found: {explicit}")

    lang_base = _find_lang_base_dir(input_dir)
    if not lang_base:
        raise FileNotFoundError("Could not detect <Language>Phrasebook folder from input_dir.")

    lang_name = lang_base.name.replace("Phrasebook", "")
    candidates = [
        f"{lang_name.lower()}_music_background.mp3",
        getattr(cfg, "MUSIC_FILENAME", ""),
    ]
    for name in candidates:
        if not name:
            continue
        candidate = lang_base / name
        if candidate.exists():
            return candidate
        if candidate.suffix == "":
            candidate = candidate.with_suffix(".mp3")
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Background music not found in {lang_base}")


def _mix_one(src: Path, music: AudioSegment, music_gain_db: float, out_dir: Path) -> None:
    dest = out_dir / src.name
    if dest.exists():
        return
    speech = AudioSegment.from_file(src)
    music_adj = music + music_gain_db
    if music_adj.duration_seconds <= 0:
        raise RuntimeError("Background music has zero duration.")
    loops = int(speech.duration_seconds // music_adj.duration_seconds) + 1
    music_looped = (music_adj * loops)[: len(speech)]
    mixed = speech.overlay(music_looped)
    mixed.export(dest, format="mp3", bitrate="192k")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Add background music to chapter audio files in a folder."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing chapter audio files (e.g., bilingual_sentences_chapters).",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Output directory. Default: <input-dir>_background",
    )
    parser.add_argument(
        "--music",
        default="",
        help="Optional path to background music file. If omitted, auto-detect from language folder.",
    )
    parser.add_argument(
        "--gain-db",
        default=None,
        type=float,
        help="Background music gain in dB (default: cfg.MUSIC_GAIN_DB or -20).",
    )
    parser.add_argument(
        "--workers",
        default=0,
        type=int,
        help="Parallel workers (default: min(cpu//2, file count)).",
    )
    args = parser.parse_args()

    input_dir = _resolve_input_dir(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else input_dir.with_name(input_dir.name + "_background")
    output_dir.mkdir(parents=True, exist_ok=True)

    music_path = _resolve_music_path(input_dir, args.music or None)
    music = AudioSegment.from_file(music_path)

    gain_db = args.gain_db
    if gain_db is None:
        gain_db = float(getattr(cfg, "MUSIC_GAIN_DB", -20.0))

    files = sorted([p for p in input_dir.glob("*.mp3")], key=lambda p: p.name.lower())
    if not files:
        print(f"No mp3 files found in {input_dir}")
        return 0

    cpu = os.cpu_count() or 1
    workers = int(args.workers) if args.workers and args.workers > 0 else max(1, cpu // 2)
    workers = min(workers, len(files))

    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Music: {music_path}")
    print(f"Gain dB: {gain_db}")
    print(f"Files: {len(files)} workers={workers}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_mix_one, f, music, gain_db, output_dir): f for f in files}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                fut.result()
            except Exception as e:
                print(f"Error {f.name}: {e}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
