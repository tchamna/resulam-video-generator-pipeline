#!/usr/bin/env python3
"""
Generate a zero-install HTML page to compare original vs denoised audio side-by-side.
"""
from __future__ import annotations

import argparse
import html
from pathlib import Path


AUDIO_EXTS = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}


def _collect(dir_path: Path) -> dict[str, Path]:
    files = {}
    for p in sorted(dir_path.iterdir()):
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files[p.name] = p
    return files


def _file_url(path: Path) -> str:
    return path.resolve().as_uri()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Original audio folder")
    parser.add_argument("--denoised", required=True, help="Denoised audio folder")
    parser.add_argument("--output", default="tools/audio_compare.html", help="Output HTML file")
    args = parser.parse_args()

    original_dir = Path(args.original)
    denoised_dir = Path(args.denoised)
    output_path = Path(args.output)

    if not original_dir.is_dir():
        raise SystemExit(f"Original folder not found: {original_dir}")
    if not denoised_dir.is_dir():
        raise SystemExit(f"Denoised folder not found: {denoised_dir}")

    orig_files = _collect(original_dir)
    den_files = _collect(denoised_dir)

    rows = []
    for name, orig_path in orig_files.items():
        den_path = den_files.get(name)
        rows.append(
            {
                "name": name,
                "orig": _file_url(orig_path),
                "den": _file_url(den_path) if den_path else None,
            }
        )

    missing = [name for name in orig_files.keys() if name not in den_files]

    html_out = [
        "<!doctype html>",
        "<html lang=\"en\">",
        "<head>",
        "<meta charset=\"utf-8\">",
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">",
        "<title>Audio Compare</title>",
        "<style>",
        ":root { --bg:#0f141a; --panel:#18212a; --text:#e6edf3; --muted:#93a4b5; --accent:#66c2a5; }",
        "body { margin:0; font-family: 'Segoe UI', Tahoma, sans-serif; background:var(--bg); color:var(--text); }",
        "header { padding:20px 24px; border-bottom:1px solid #243241; }",
        "h1 { margin:0 0 6px; font-size:20px; }",
        "p { margin:4px 0; color:var(--muted); }",
        ".wrap { padding:16px 24px 28px; }",
        ".row { display:grid; grid-template-columns: 1.3fr 1fr 1fr 220px; gap:16px; align-items:center; padding:10px 12px; background:var(--panel); border-radius:10px; margin-bottom:10px; }",
        ".name { font-size:14px; word-break:break-all; }",
        "audio { width:100%; }",
        ".btns { display:flex; gap:8px; }",
        "button { background:#223040; color:var(--text); border:1px solid #2b3c4d; padding:6px 10px; border-radius:6px; cursor:pointer; }",
        "button:hover { border-color:var(--accent); color:var(--accent); }",
        ".missing { color:#ffb3b3; font-size:12px; }",
        ".meta { font-size:12px; color:var(--muted); }",
        "</style>",
        "</head>",
        "<body>",
        "<header>",
        "<h1>Audio Compare</h1>",
        f"<p>Original: {html.escape(str(original_dir))}</p>",
        f"<p>Denoised: {html.escape(str(denoised_dir))}</p>",
        "</header>",
        "<div class=\"wrap\">",
    ]

    if missing:
        html_out.append(
            f"<p class=\"missing\">Missing denoised files: {html.escape(', '.join(missing))}</p>"
        )

    for idx, row in enumerate(rows):
        name = html.escape(row["name"])
        orig = row["orig"]
        den = row["den"]
        den_label = "Missing" if den is None else "Denoised"
        den_audio = (
            f"<audio id=\"den-{idx}\" controls src=\"{den}\"></audio>"
            if den
            else "<span class=\"missing\">Missing</span>"
        )
        html_out.extend(
            [
                "<div class=\"row\">",
                f"<div class=\"name\">{name}</div>",
                f"<div><div class=\"meta\">Original</div><audio id=\"orig-{idx}\" controls src=\"{orig}\"></audio></div>",
                f"<div><div class=\"meta\">{den_label}</div>{den_audio}</div>",
                "<div class=\"btns\">",
                f"<button onclick=\"playA({idx})\">Play A</button>",
                f"<button onclick=\"playB({idx})\">Play B</button>",
                f"<button onclick=\"stopAll({idx})\">Stop</button>",
                "</div>",
                "</div>",
            ]
        )

    html_out.extend(
        [
            "</div>",
            "<script>",
            "function _get(id){ return document.getElementById(id); }",
            "function stopAll(i){",
            "  const a=_get(`orig-${i}`); const b=_get(`den-${i}`);",
            "  if(a){a.pause(); a.currentTime=0;}",
            "  if(b){b.pause(); b.currentTime=0;}",
            "}",
            "function playA(i){",
            "  const a=_get(`orig-${i}`); const b=_get(`den-${i}`);",
            "  if(b){b.pause();}",
            "  if(a){a.currentTime=0; a.play();}",
            "}",
            "function playB(i){",
            "  const a=_get(`orig-${i}`); const b=_get(`den-${i}`);",
            "  if(a){a.pause();}",
            "  if(b){b.currentTime=0; b.play();}",
            "}",
            "</script>",
            "</body>",
            "</html>",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(html_out), encoding="utf-8")
    print(f"Wrote: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
