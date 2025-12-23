#!/usr/bin/env python3
"""
AI-powered noise removal using DeepFilterNet (df CLI or API fallback).

Example:
  python tools/ai_denoise.py --input "assets/audio" --output "assets/denoised"
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def _find_df_cli() -> list[str] | None:
    df_path = shutil.which("df")
    if df_path:
        return [df_path]
    return None


def _can_use_df_api() -> bool:
    try:
        enhance = importlib.import_module("df.enhance")
        df_io = importlib.import_module("df.io")
        has_enhance = hasattr(enhance, "init_df") and hasattr(enhance, "enhance")
        has_io = hasattr(df_io, "load_audio") and hasattr(df_io, "save_audio")
        return bool(has_enhance and has_io)
    except Exception:
        return False


def _can_use_demucs() -> bool:
    return importlib.util.find_spec("demucs") is not None


def _ffmpeg_available() -> bool:
    return bool(shutil.which("ffmpeg"))


def _run(cmd: list[str], *, verbose: bool) -> None:
    if verbose:
        print(" ".join(cmd))
    subprocess.run(cmd, check=True, text=True, capture_output=not verbose)


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _collect_inputs(path: Path, *, recursive: bool) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    files = [p for p in path.glob(pattern) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def _df_cli_supports_short_flags(df_cmd: list[str]) -> bool:
    try:
        run = subprocess.run(df_cmd + ["--help"], text=True, capture_output=True, check=False)
        help_text = (run.stdout or "") + (run.stderr or "")
    except Exception:
        return True
    if "-i" in help_text and "-o" in help_text:
        return True
    if "--input" in help_text or "--output" in help_text:
        return False
    return True


def _df_cli_denoise(
    df_cmd: list[str],
    input_wav: Path,
    output_wav: Path,
    *,
    verbose: bool,
) -> None:
    short_flags = _df_cli_supports_short_flags(df_cmd)
    if short_flags:
        cmd = df_cmd + ["-i", str(input_wav), "-o", str(output_wav)]
    else:
        cmd = df_cmd + ["--input", str(input_wav), "--output", str(output_wav)]
    _run(cmd, verbose=verbose)


def _df_api_denoise(
    input_wav: Path,
    output_wav: Path,
    *,
    model=None,
    df_state=None,
    atten_lim_db: float | None = None,
    dry_wet: float = 1.0,
) -> None:
    enhance = importlib.import_module("df.enhance")
    df_io = importlib.import_module("df.io")
    load_audio = getattr(df_io, "load_audio", None)
    save_audio = getattr(df_io, "save_audio", None)
    if load_audio is None or save_audio is None:
        raise RuntimeError("DeepFilterNet API changed; cannot find load/save helpers.")
    if model is None or df_state is None:
        model, df_state, _ = enhance.init_df()
    audio, _meta = load_audio(str(input_wav), sr=df_state.sr(), verbose=False)
    enhanced = enhance.enhance(model, df_state, audio, atten_lim_db=atten_lim_db)
    if dry_wet < 1.0:
        enhanced = (audio * (1.0 - dry_wet)) + (enhanced * dry_wet)
    save_audio(str(output_wav), enhanced, df_state.sr())


def _convert_to_wav(src: Path, dst: Path, *, verbose: bool) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "48000",
        str(dst),
    ]
    _run(cmd, verbose=verbose)


def _blend_audio(
    original_wav: Path,
    processed_wav: Path,
    output_wav: Path,
    *,
    dry_wet: float,
    verbose: bool,
) -> None:
    dry_wet = max(0.0, min(1.0, dry_wet))
    dry = 1.0 - dry_wet
    wet = dry_wet
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(original_wav),
        "-i",
        str(processed_wav),
        "-filter_complex",
        f"[0:a]volume={dry:.3f}[a0];[1:a]volume={wet:.3f}[a1];[a0][a1]amix=inputs=2:weights={dry:.3f} {wet:.3f}:normalize=0",
        "-ac",
        "1",
        "-ar",
        "48000",
        str(output_wav),
    ]
    _run(cmd, verbose=verbose)


def _demucs_denoise(
    input_wav: Path,
    output_wav: Path,
    *,
    model: str,
    stem: str,
    dry_wet: float,
    verbose: bool,
) -> None:
    tmp_dir = output_wav.parent / "_demucs_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "demucs",
        "--two-stems",
        stem,
        "--name",
        model,
        "-o",
        str(tmp_dir),
        str(input_wav),
    ]
    _run(cmd, verbose=verbose)
    stem_path = tmp_dir / model / input_wav.stem / f"{stem}.wav"
    if not stem_path.exists():
        raise RuntimeError(f"Demucs output not found: {stem_path}")
    if dry_wet < 1.0:
        _blend_audio(input_wav, stem_path, output_wav, dry_wet=dry_wet, verbose=verbose)
    else:
        shutil.copy2(stem_path, output_wav)
def _apply_declick(src: Path, dst: Path, *, strength: float, verbose: bool) -> None:
    # strength in [0,1]; mapped to adeclick defaults
    strength = _clamp(strength, 0.0, 1.0)
    window = int(round(10 + (strength * 90)))
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-af",
        f"adeclick=w={window}",
        str(dst),
    ]
    _run(cmd, verbose=verbose)


def _apply_declick_params(
    src: Path,
    dst: Path,
    *,
    window: int,
    overlap: float,
    arorder: int,
    threshold: float,
    burst: float,
    method: str,
    verbose: bool,
) -> None:
    window = int(round(_clamp(float(window), 10, 100)))
    overlap = _clamp(float(overlap), 50, 95)
    arorder = int(round(_clamp(float(arorder), 0, 25)))
    threshold = _clamp(float(threshold), 1, 100)
    burst = _clamp(float(burst), 0, 10)
    method = method.lower()
    if method not in {"add", "save"}:
        method = "add"
    af = (
        f"adeclick=w={window}:o={overlap:.2f}:a={arorder}:t={threshold:.2f}:b={burst:.2f}:m={method}"
    )
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-af",
        af,
        str(dst),
    ]
    _run(cmd, verbose=verbose)


def _convert_from_wav(src: Path, dst: Path, *, verbose: bool) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        str(dst),
    ]
    _run(cmd, verbose=verbose)


def _build_output_path(
    input_file: Path,
    output_dir: Path,
    *,
    output_format: str | None,
) -> Path:
    if output_format:
        ext = output_format if output_format.startswith(".") else f".{output_format}"
        return output_dir / f"{input_file.stem}{ext}"
    return output_dir / input_file.name


def main() -> int:
    parser = argparse.ArgumentParser(description="AI noise removal using DeepFilterNet.")
    parser.add_argument("--input", required=True, help="Input file or folder")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--recursive", action="store_true", help="Process subfolders")
    parser.add_argument("--format", help="Force output format (e.g., wav, mp3)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    parser.add_argument(
        "--backend",
        choices=["deepfilternet", "demucs"],
        default="deepfilternet",
        help="Denoise backend",
    )
    parser.add_argument(
        "--atten-lim-db",
        type=float,
        default=None,
        help="Attenuation limit in dB (lower magnitude preserves voice more)",
    )
    parser.add_argument(
        "--dry-wet",
        type=float,
        default=1.0,
        help="Blend denoised signal with original (0.0-1.0)",
    )
    parser.add_argument(
        "--demucs-model",
        default="htdemucs",
        help="Demucs model name (e.g., htdemucs, htdemucs_ft)",
    )
    parser.add_argument(
        "--demucs-stem",
        default="vocals",
        help="Demucs stem to keep (default: vocals)",
    )
    parser.add_argument(
        "--declick",
        action="store_true",
        help="Run a light declick pass before denoising",
    )
    parser.add_argument(
        "--declick-strength",
        type=float,
        default=0.4,
        help="Declick strength (0.0-1.0)",
    )
    parser.add_argument(
        "--declick-profile",
        choices=["light", "medium", "strong"],
        help="Declick profile preset",
    )
    parser.add_argument("--declick-window", type=int, help="Declick window size (10-100)")
    parser.add_argument("--declick-overlap", type=float, help="Declick overlap (50-95)")
    parser.add_argument("--declick-arorder", type=int, help="Declick AR order (0-25)")
    parser.add_argument("--declick-threshold", type=float, help="Declick threshold (1-100)")
    parser.add_argument("--declick-burst", type=float, help="Declick burst (0-10)")
    parser.add_argument(
        "--declick-method",
        choices=["add", "save"],
        help="Declick overlap method",
    )
    parser.add_argument("--verbose", action="store_true", help="Print executed commands")
    args = parser.parse_args()

    in_path = Path(args.input).expanduser()
    out_dir = Path(args.output).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = _collect_inputs(in_path, recursive=args.recursive)
    if not inputs:
        print(f"No audio files found in: {in_path}")
        return 2

    df_cmd = None
    use_api = False
    use_demucs = args.backend == "demucs"
    if use_demucs:
        if not _can_use_demucs():
            print("Demucs not found.")
            print("Install with: pip install demucs")
            return 3
    else:
        df_cmd = _find_df_cli()
        if not df_cmd:
            use_api = _can_use_df_api()
        if not df_cmd and not use_api:
            print("DeepFilterNet not found.")
            print("Install with: pip install deepfilternet")
            return 3

    if not _ffmpeg_available():
        print("ffmpeg is required for format conversion.")
        return 4

    processed = 0
    skipped = 0
    df_model = None
    df_state = None
    if use_api:
        enhance = importlib.import_module("df.enhance")
        df_model, df_state, _ = enhance.init_df()
    for src in inputs:
        out_path = _build_output_path(src, out_dir, output_format=args.format)
        if out_path.exists() and not args.overwrite:
            skipped += 1
            continue

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir = Path(tmp_dir)
            tmp_in = tmp_dir / "in.wav"
            tmp_clean = tmp_dir / "clean.wav"
            tmp_out = tmp_dir / "out.wav"

            _convert_to_wav(src, tmp_in, verbose=args.verbose)
            denoise_in = tmp_in
            if args.declick:
                profile_map = {
                    "light": dict(window=35, overlap=70, arorder=2, threshold=3.0, burst=2.0, method="add"),
                    "medium": dict(window=55, overlap=80, arorder=4, threshold=2.0, burst=3.0, method="add"),
                    "strong": dict(window=85, overlap=90, arorder=8, threshold=1.5, burst=5.0, method="add"),
                }
                params = None
                if args.declick_profile:
                    params = profile_map.get(args.declick_profile)
                if params:
                    _apply_declick_params(tmp_in, tmp_clean, verbose=args.verbose, **params)
                elif any(
                    v is not None
                    for v in (
                        args.declick_window,
                        args.declick_overlap,
                        args.declick_arorder,
                        args.declick_threshold,
                        args.declick_burst,
                        args.declick_method,
                    )
                ):
                    _apply_declick_params(
                        tmp_in,
                        tmp_clean,
                        window=args.declick_window if args.declick_window is not None else 55,
                        overlap=args.declick_overlap if args.declick_overlap is not None else 75,
                        arorder=args.declick_arorder if args.declick_arorder is not None else 2,
                        threshold=args.declick_threshold if args.declick_threshold is not None else 2.0,
                        burst=args.declick_burst if args.declick_burst is not None else 2.0,
                        method=args.declick_method or "add",
                        verbose=args.verbose,
                    )
                else:
                    _apply_declick(
                        tmp_in,
                        tmp_clean,
                        strength=args.declick_strength,
                        verbose=args.verbose,
                    )
                denoise_in = tmp_clean

            if use_demucs:
                _demucs_denoise(
                    denoise_in,
                    tmp_out,
                    model=args.demucs_model,
                    stem=args.demucs_stem,
                    dry_wet=args.dry_wet,
                    verbose=args.verbose,
                )
            elif df_cmd:
                _df_cli_denoise(df_cmd, denoise_in, tmp_out, verbose=args.verbose)
            else:
                _df_api_denoise(
                    denoise_in,
                    tmp_out,
                    model=df_model,
                    df_state=df_state,
                    atten_lim_db=args.atten_lim_db,
                    dry_wet=args.dry_wet,
                )

            if out_path.suffix.lower() == ".wav":
                shutil.copy2(tmp_out, out_path)
            else:
                _convert_from_wav(tmp_out, out_path, verbose=args.verbose)

        processed += 1
        print(f"OK: {src.name} -> {out_path.name}")

    print(f"Done. processed={processed} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
