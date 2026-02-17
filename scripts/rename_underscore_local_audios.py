from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _infer_language_from_local_dir(local_dir: Path) -> str | None:
    # Try to infer from .../Languages/<Lang>Phrasebook/<Lang>Only[/...]
    parts = list(local_dir.resolve().parts)
    try:
        i = parts.index("Languages")
    except ValueError:
        return None
    if i + 1 >= len(parts):
        return None
    phrasebook = parts[i + 1]
    if not phrasebook.endswith("Phrasebook"):
        return None
    return phrasebook[: -len("Phrasebook")] or None


def _find_underscore_audio_files(local_dir: Path) -> list[Path]:
    files: list[Path] = []
    for ext in ("*.mp3", "*.wav", "*.m4a", "*.aac"):
        files.extend(local_dir.rglob(ext))
    files = [p for p in files if p.is_file() and p.stem.endswith("_")]
    files.sort(key=lambda p: p.as_posix().lower())
    return files


_ID_RE = re.compile(r"(\d+)_$")


def _extract_id(p: Path) -> int | None:
    m = _ID_RE.search(p.stem)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Recursively find local audio files ending with '_' (e.g., '66_.mp3') and copy/rename them to "
            "'<language>_phrasebook_<id><ext>' under Results_Audios/normal_rythm."
        )
    )
    ap.add_argument("--language", default="", help="Language name, e.g. Yemba. If omitted, inferred from --local-dir.")
    ap.add_argument(
        "--local-dir",
        default="",
        help="Path to <Lang>Only folder (e.g., private_assets/Languages/YembaPhrasebook/YembaOnly).",
    )
    ap.add_argument(
        "--base-dir",
        default=str(REPO_ROOT),
        help="Repo root (default: detected from this script location).",
    )
    ap.add_argument(
        "--use-private-assets",
        action="store_true",
        default=True,
        help="Use private_assets (default).",
    )
    ap.add_argument(
        "--use-assets",
        action="store_true",
        default=False,
        help="Use assets instead of private_assets.",
    )
    ap.add_argument(
        "--output-dir",
        default="",
        help="Optional explicit output directory. Default: <assets>/Languages/<Lang>Phrasebook/Results_Audios/normal_rythm",
    )
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = ap.parse_args()

    base_dir = Path(args.base_dir).resolve()
    use_private = True
    if args.use_assets:
        use_private = False
    assets_root = base_dir / ("private_assets" if use_private else "assets")

    language = (args.language or "").strip()

    if args.local_dir:
        local_dir = Path(args.local_dir)
        if not local_dir.is_absolute():
            local_dir = (base_dir / args.local_dir).resolve()
    else:
        if not language:
            print("Error: provide --language or --local-dir", file=sys.stderr)
            return 2
        local_dir = (assets_root / f"Languages/{language.title()}Phrasebook/{language.title()}Only").resolve()

    if not local_dir.exists():
        print(f"Error: local dir not found: {local_dir}", file=sys.stderr)
        return 2

    if not language:
        inferred = _infer_language_from_local_dir(local_dir)
        if not inferred:
            print("Error: could not infer language from --local-dir; pass --language", file=sys.stderr)
            return 2
        language = inferred

    if args.output_dir:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = (base_dir / args.output_dir).resolve()
    else:
        out_dir = assets_root / f"Languages/{language.title()}Phrasebook/Results_Audios/normal_rythm"
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = _find_underscore_audio_files(local_dir)
    if not candidates:
        print(f"No underscore audio files found under: {local_dir}")
        return 0

    written = 0
    skipped = 0
    errors = 0

    for src in candidates:
        sid = _extract_id(src)
        if sid is None:
            skipped += 1
            continue
        dest = out_dir / f"{language.lower()}_phrasebook_{sid}{src.suffix.lower()}"
        if dest.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            shutil.copy2(src, dest)
            written += 1
        except Exception as e:
            errors += 1
            print(f"Error copying {src} -> {dest}: {e}", file=sys.stderr)

    print(f"Language={language.title()}")
    print(f"Local dir={local_dir}")
    print(f"Output dir={out_dir}")
    print(f"Found={len(candidates)} written={written} skipped={skipped} errors={errors}")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
