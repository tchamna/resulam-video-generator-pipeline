from __future__ import annotations
"""
multi_language_video_generator_v2.py  • 2025‑07‑21 (k6)
────────────────────────────────────────────────────────────────────────────
• LANGUAGE switch: "Nufi", "Yoruba", …
• One background per chapter (detects “Chapter …” lines)
• CPU‑aware parallelism (FFMPEG_THREADS_PER_JOB × PYTHON_JOBS)
• Header layout
      <Language>   ……   “Please Support Resulam”
      sentence‑number        (next row, right‑aligned)
• Footer URL call left in place but commented out
• Logos resized to 100 px
• 👉 **Fully migrated to MoviePy v2.x API (with_ / resized / with_audio)**
"""

import os
import shutil
import threading
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageFile
from moviepy import (
    AudioFileClip,
    CompositeAudioClip,
    CompositeVideoClip,
    ImageClip,
    TextClip,
)

# ───────────────────────────── PARALLELISM ───────────────────────────────
FFMPEG_THREADS_PER_JOB = 4           # threads used *inside* one ffmpeg encode
PYTHON_JOBS            = 5           # concurrent MoviePy encodes

cores = os.cpu_count() or 1
while FFMPEG_THREADS_PER_JOB > cores:
    FFMPEG_THREADS_PER_JOB //= 2 or 1
PYTHON_JOBS = min(PYTHON_JOBS, max(1, cores // FFMPEG_THREADS_PER_JOB))

# ───────────────────────────── USER SETTINGS ─────────────────────────────
LANGUAGE = "Yoruba"      # ← quick switch here (e.g. "Nufi")

ROOT      = Path(r"G:/My Drive/Data_Science/Resulam/Phrasebook_Audio_Video_Processing_production")
FONT_PATH = Path(r"C:/Users/tcham/OneDrive/Documents/Workspace_Codes/dictionnaire-nufi-franc-nufi/app/src/main/assets/fonts/CharisSIL-B.ttf")
LOGO_PATH = Path(r"G:/My Drive/Data_Science/Resulam/Phrasebook_Audio_Video_Processing_production_Backup/resulam_logo_resurrectionLangue.png")

VIDEO_SIZE        = (1920, 1080)
FPS               = 24
PAUSE_BTWN_REPEAT = 5
FINAL_PAUSE       = 3
BASE_FS           = 100
BANNER_GAP_RATIO  = 0.30
RIGHT_MARGIN_PX   = 25
TOP_MARGIN_PX     = 15

INTRO_LINES = {
    "Nufi":    "Yū' Mfʉ́ɑ́'sí, Mfāhngə́ə́:",
    "Swahili": "Sikiliza, rudia na tafsiri:",
    "Yoruba":  "Tẹ́tí, tunsọ, ṣe ògbùfọ̀:",
}
DEFAULT_INTRO = "Listen, repeat and translate:"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ───────────────────────────── PATH HELPERS ──────────────────────────────

def build_paths(lang: str):
    title, slug = lang, lang.lower()
    return {
        "language": title,
        "slug": slug,
        "bg_dir":   ROOT / "Backgrounds_Selected",
        "nufi_dir": ROOT / f"Languages/{title}Phrasebook/{title}Only",
        "eng_dir":  ROOT / "EnglishOnly",
        "out_dir":  ROOT / f"Python_Scripts_Resulam_Phrasebooks_Audio_Processing/{title}Homework",
        "sent_txt": ROOT / f"Languages/{title}Phrasebook/{slug}_english_french_phrasebook_sentences_list.txt",
    }


def list_backgrounds(folder: Path):
    files = [p for ext in ("*.png", "*.jpg", "*.jpeg") for p in folder.glob(ext)]
    if not files:
        raise RuntimeError(f"No background images in {folder}")
    return files

# ───────────────────────────── SENTENCE PARSER ───────────────────────────

def parse_sentences(txt: Path, slug: str):
    rows = []
    with open(txt, encoding="utf-8") as fh:
        for ln in fh:
            if "|" not in ln:
                continue
            try:
                id_part, en_raw = ln.split(")", 1)
                sid = int(id_part.strip())
                _, src, fr = [x.strip() for x in ln.strip().split("|", 2)]
                en = en_raw.split("|", 1)[0].strip()
                rows.append({
                    "id": sid,
                    "source": src,
                    "english": en,
                    "french": fr,
                    "nufi_mp3": f"{slug}_phrasebook_{sid}.mp3",
                    "eng_mp3":  f"english_{sid}.mp3",
                })
            except ValueError:
                print(f"⚠ bad line: {ln.strip()}")
    return rows

# ───────────────────────────── CHAPTER RANGES ────────────────────────────

def chapter_ranges(sents):
    ranges, start = [], None
    for s in sorted(sents, key=lambda x: x["id"]):
        if s["english"].lower().startswith("chapter"):
            if start is not None:
                ranges.append((start, prev))
            start = s["id"]
        prev = s["id"]
    if start is not None:
        ranges.append((start, prev))
    return ranges

# ───────────────────────────── BACKGROUND UTILS ──────────────────────────

def ensure_bg(img: Path, target=VIDEO_SIZE):
    with Image.open(img) as im:
        if im.size == target:
            return img
        even = (target[0] - target[0] % 2, target[1] - target[1] % 2)
        out  = img.with_stem(img.stem + "_resized")
        im.resize(even, Image.LANCZOS).save(out)
        return out

def tag_bgs(rows, bgs):
    idx, current = -1, bgs[0]
    output = []
    for r in sorted(rows, key=lambda x: x["id"]):
        if r["english"].lower().startswith("chapter"):
            idx = (idx + 1) % len(bgs)
            current = bgs[idx]
        rec = dict(r)
        rec["bg"] = current
        output.append(rec)
    return output

# ───────────────────────────── CAPTIONS ──────────────────────────────────

def font_size(row):
    n = len(f"{row['source']} {row['english']} {row['french']}")
    return BASE_FS if n < 50 else int(BASE_FS * (0.85 if n < 80 else 0.75 if n < 110 else 0.65))

def add_cap(clips, txt, start, dur, y, color, fs, wrap=True, align="center"):
    mode = "caption" if wrap else "label"
    size = (VIDEO_SIZE[0] - 200, None) if wrap else None
    clip = (
        TextClip(str(FONT_PATH), text=txt, font_size=fs, 
                 color=color, method=mode, size=size)
        .with_position((align, y))
        .with_start(start)
        .with_duration(dur)
    )
    clips.append(clip)
    return y + clip.h + int(fs * 0.20)

# ───────────────────────────── VIDEO BUILDER ─────────────────────────────

def build_video(row, paths):
    out = paths["out_dir"] / f"{paths['slug']}_sentence_{row['id']}.mp4"
    if out.exists():
        return

    nufi_path = paths["nufi_dir"] / row["nufi_mp3"]
    eng_path  = paths["eng_dir"]  / row["eng_mp3"]
    if not nufi_path.exists() or not eng_path.exists():
        print(f"⚠ audio missing {row['id']}")
        return

    nufi = AudioFileClip(str(nufi_path))
    eng  = AudioFileClip(str(eng_path))

    d    = nufi.duration
    t2   = d + PAUSE_BTWN_REPEAT
    t3   = t2 + d + PAUSE_BTWN_REPEAT
    stE  = t3 + d + PAUSE_BTWN_REPEAT
    total = stE + eng.duration + FINAL_PAUSE

    audio = CompositeAudioClip([
        nufi.copy().with_start(0),
        nufi.copy().with_start(t2),
        nufi.copy().with_start(t3),
        eng.copy().with_start(stE),
    ])

    fs = font_size(row)
    clips = [ImageClip(str(row["bg"]) ).with_duration(total)]

    clips.append(
        TextClip(str(FONT_PATH), text=paths["language"], font_size=int(fs*0.55), color="yellow", method="label")
        .with_position((15, TOP_MARGIN_PX))
        .with_duration(total)
    )
    support = (
        TextClip(str(FONT_PATH), text="Please Support Resulam", font_size=int(fs*0.5), color="yellow", method="label")
        .with_position(("right", TOP_MARGIN_PX))
        .with_duration(total)
    )
    clips.append(support)

    num_clip = (
        TextClip(str(FONT_PATH), text=str(row["id"]), font_size=int(fs*0.6), color="white", method="label")
        .with_position((VIDEO_SIZE[0]-RIGHT_MARGIN_PX, TOP_MARGIN_PX+support.h+int(fs*0.15)))
        .with_duration(total)
    )
    clips.append(num_clip)

    first_y = TOP_MARGIN_PX + max(num_clip.h, support.h) + int(fs * BANNER_GAP_RATIO)
    y = first_y

    y = add_cap(clips, INTRO_LINES.get(paths["language"], DEFAULT_INTRO), 0, t2, y, "white", int(fs*0.9))
    y = add_cap(clips, DEFAULT_INTRO, 0, t2, y, "yellow", fs)

    y = first_y
    y = add_cap(clips, row["source"],  t2, total-t2, y, "white", fs)
    y = add_cap(clips, row["english"], t3, total-t3, y, "yellow", fs)
    add_cap(clips, row["french"],      t3, total-t3, y, "white", fs)

    # Footer (commented)
    # add_cap(clips, "www.resulam.com", 0, total, VIDEO_SIZE[1]-95, "yellow", int(fs*0.5), wrap=False)

    for pos in ("left", "right"):
        clips.append(
            ImageClip(str(LOGO_PATH)).resized(height=100)
            .with_position((pos, "bottom"))
            .with_duration(total)
        )

    tmp_aac = f"tmp-{uuid4().hex}.m4a"
    tmp_mp4 = out.with_suffix(".tmp.mp4")
    try:
        (
            CompositeVideoClip(clips, size=VIDEO_SIZE)
            .with_audio(audio)
            .with_duration(total)
            .write_videofile(
                str(tmp_mp4), fps=FPS, codec="libx264", audio_codec="aac",
                temp_audiofile=tmp_aac, remove_temp=True,
                ffmpeg_params=["-pix_fmt","yuv420p","-profile:v","high","-level","4.1","-movflags","+faststart"],
                preset="ultrafast", threads=FFMPEG_THREADS_PER_JOB, 
                
            )
        )
        shutil.move(tmp_mp4, out)
        print(f"✅  {out.name}")
    except Exception as err:
        print(f"❌ id={row['id']}  {err}")
        if tmp_mp4.exists(): tmp_mp4.unlink()

# ───────────────────────────── THREAD WORKER & MAIN ────────────────────────

sem = threading.Semaphore(PYTHON_JOBS)

def render_slice(rows, start_id, end_id, paths):
    with sem:
        for r in rows:
            if start_id <= r['id'] <= end_id:
                build_video(r, paths)

def process_language(lang: str):
    paths = build_paths(lang)
    raw   = parse_sentences(paths['sent_txt'], paths['slug'])
    ranges= chapter_ranges(raw)
    bgs   = [ensure_bg(p) for p in list_backgrounds(paths['bg_dir'])]
    rows  = tag_bgs(raw, bgs)

    threads = []
    for a,b in ranges:
        t = threading.Thread(target=render_slice, args=(rows,a,b,paths))
        t.start(); threads.append(t)
    for t in threads: t.join()
    print(f"✔ Completed {lang}")

if __name__ == '__main__':
    process_language(LANGUAGE)
