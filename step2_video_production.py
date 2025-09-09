"""
multi_language_video_generator.py  • single build_video(mode)
────────────────────────────────────────────────────────────────────────────
• LANGUAGE switch: "Nufi", "Yoruba", "Duala", …
• One background per chapter (detects “Chapter …” lines)
• CPU-aware parallelism (FFMPEG_THREADS_PER_JOB × PYTHON_JOBS)
• Header layout
      <Language>   ……   “Please Support Resulam”
      sentence-number        (next row, right-aligned)
• Logos resized to 100 px
• Modes:
   - homework: intro + 3× local repeats → English  → saves under .../<Language>/Homework
   - lecture : all texts visible; English → pause → local → saves under .../<Language>/Lecture

Usage:
   # Set MODE = "lecture" or "homework" below, then run this file.
"""

from __future__ import annotations
import os, sys, shutil, threading
from pathlib import Path
from uuid import uuid4
from typing import List, Dict
from PIL import Image, ImageFile
from moviepy.editor import (
    AudioFileClip, CompositeAudioClip, CompositeVideoClip,
    ImageClip, TextClip,
)

# ── USER SELECTIONS ──────────────────────────────────────────────────────
LANGUAGE = "Duala"         # e.g., "Nufi", "Yoruba", "Duala", "Swahili", "Fe'efe'e"
MODE     = "lecture"       # "lecture" or "homework"
# MODE     = "homework"       # "lecture" or "homework"

# ── PARALLELISM SETTINGS ────────────────────────────────────────────────
FFMPEG_THREADS_PER_JOB = 4
PYTHON_JOBS            = 5

logical = os.cpu_count() or 1
while FFMPEG_THREADS_PER_JOB > logical:
    FFMPEG_THREADS_PER_JOB //= 2 or 1
PYTHON_JOBS = min(PYTHON_JOBS, max(1, logical // FFMPEG_THREADS_PER_JOB))

# ── BASIC PATH / FONT SETTINGS ───────────────────────────────────────────
ROOT      = Path(r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production")
# FONT_PATH = Path(r"C:\Users\tcham\OneDrive\Documents\Workspace_Codes\dictionnaire-nufi-franc-nufi"
#                  r"\app\src\main\assets\fonts\CharisSIL-B.ttf")

# FONT_PATH = Path(r"C:\Windows\Fonts\arialbd.ttf")
FONT_PATH = ROOT/Path(r"Fonts\arialbd.ttf")
LOGO_PATH = ROOT/Path(r"resulam_logo_resurrectionLangue.png")

VIDEO_SIZE        = (1920, 1080)
FPS               = 24
PAUSE_BTWN_REPEAT = 5
FINAL_PAUSE       = 3
BASE_FS           = 100
BANNER_GAP_RATIO  = 0.30
RIGHT_MARGIN_PX   = 25
TOP_MARGIN_PX     = 15

# PROD_OR_TEST = "Production"  # or "Test"
PROD_OR_TEST = "Test"  # or "Test"

if PROD_OR_TEST == "Test":
    PROD_OR_TEST = "Test"
else:
    PROD_OR_TEST = ""

INTRO_LINES = {
    "Nufi":    "Yū' Mfʉ́ɑ́'sí, Mfāhngə́ə́:",
    "Swahili": "Sikiliza, rudia na tafsiri:",
    "Yoruba":  "Tẹ́tí gbọ́, tunsọ, ṣe ògbùfọ̀:",
    "Duala":   "Seŋgâ, Timbísɛ́lɛ̂ na Túkwâ:",
}
DEFAULT_INTRO = "Listen, repeat and translate:"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── PATH HELPERS ─────────────────────────────────────────────────────────
def build_paths(lang: str, mode: str) -> Dict[str, Path | str]:
    """Return standardized paths; write outputs into Lecture/Homework subfolders based on mode."""
    lang_title, lang_lower = lang.title(), lang.lower()

    # Output directory split by mode
    mode_norm = "Lecture" if mode.lower().strip() == "lecture" else "Homework"
    out_dir = ROOT / f"Python_Scripts_Resulam_Phrasebooks_Audio_Processing" / lang_title / mode_norm
    os.makedirs(out_dir, exist_ok=True)

    # Background fallback: <Language> → Common → parent folder
    cand = ROOT / "Backgrounds_Selected" / lang_title
    if cand.exists():
        bg_dir = cand
    else:
        common = ROOT / "Backgrounds_Selected" / "Common"
        bg_dir = common if common.exists() else (ROOT / "Backgrounds_Selected")

    return {
        "language": lang_title,
        "lang_lower": lang_lower,
        "bg_dir": bg_dir,
        "nufi_dir": ROOT / f"Languages/{lang_title}Phrasebook/{lang_title}Only{PROD_OR_TEST}",
        "eng_dir":  ROOT / "EnglishOnly",
        "out_dir":  out_dir,
        "sent_txt": ROOT / f"Languages/{lang_title}Phrasebook/{lang_lower}_english_french_phrasebook_sentences_list.txt",
    }

def list_backgrounds(dir_: Path) -> List[Path]:
    imgs = [p for ext in ("*.png","*.jpg","*.jpeg") for p in dir_.glob(ext)]
    if not imgs:
        raise RuntimeError(f"No backgrounds in {dir_}")
    return imgs

# ── SENTENCE PARSER ──────────────────────────────────────────────────────

def parse_sentences(txt: Path, lang_lower: str):
    rows=[]
    with open(txt, encoding="utf-8") as fh:
        for ln in fh:
            if "|" not in ln: continue
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
                    "nufi_mp3": f"{lang_lower}_phrasebook_{sid}_padded.mp3",
                    "eng_mp3":  f"english_{sid}.mp3",
                })
            except ValueError:
                print(f"⚠ bad line: {ln.strip()}")
    return rows

# ── CHAPTER RANGES ───────────────────────────────────────────────────────
def chapter_ranges(sents: List[Dict]) -> List[tuple]:
    ranges, start = [], None
    prev = None
    for s in sorted(sents, key=lambda x: x["id"]):
        if s["english"].lower().startswith("chapter"):
            if start is not None and prev is not None:
                ranges.append((start, prev))
            start = s["id"]
        prev = s["id"]
    if start is not None and prev is not None:
        ranges.append((start, prev))
    if not ranges and sents:
        ids = sorted(x["id"] for x in sents)
        ranges = [(ids[0], ids[-1])]
    return ranges

# ── BACKGROUND UTILITIES ────────────────────────────────────────────────
def ensure_bg(img: Path, tgt=VIDEO_SIZE) -> Path:
    with Image.open(img) as im:
        if im.size == tgt:
            return img
        even = (tgt[0] - tgt[0] % 2, tgt[1] - tgt[1] % 2)
        try:
            out = img.with_stem(img.stem + "_resized")
        except AttributeError:
            out = img.with_name(img.stem + "_resized" + img.suffix)
        im.resize(even, Image.LANCZOS).save(out)
        return out

def tag_bgs(sents: List[Dict], bgs: List[Path]) -> List[Dict]:
    idx, cur, out = -1, bgs[0], []
    for s in sorted(sents, key=lambda x: x["id"]):
        if s["english"].lower().startswith("chapter"):
            idx = (idx + 1) % len(bgs)
            cur = bgs[idx]
        d = dict(s); d["bg"] = cur; out.append(d)
    return out

# ── CAPTION HELPERS ─────────────────────────────────────────────────────
def font_size(s: Dict) -> int:
    n = len(f"{s['source']} {s['english']} {s['french']}")
    return (BASE_FS if n < 50 else int(BASE_FS*0.85) if n < 80
            else int(BASE_FS*0.75) if n < 110 else int(BASE_FS*0.65))

def add_cap(lst: List, txt: str, start: float, dur: float,
            y: int, color: str, fs: int, wrap=True, align="center") -> int:
    mode = "caption" if wrap else "label"
    size = (VIDEO_SIZE[0]-200, None) if wrap else None
    clip = (TextClip(txt, font=str(FONT_PATH), fontsize=fs, color=color,
                     method=mode, size=size)
            .set_position((align, y)).set_start(start).set_duration(dur))
    lst.append(clip)
    return y + clip.h + int(fs*0.20)


# ── SINGLE VIDEO BUILDER (mode switch) ───────────────────────────────────
def build_video(s: Dict, p: Dict, mode: str = "lecture"):
    """
    mode="lecture": all captions visible entire duration; audio = English → pause → Local.
    mode="homework": intro + staged captions; audio = Local×3 (with pauses) → English.
    """
    out = p["out_dir"] / f"{p['lang_lower']}_sentence_{s['id']}.mp4"
    if out.exists():
        return

    nufi_path = p["nufi_dir"] / "gen2_normalized_padded"/ s["nufi_mp3"]
    eng_path  = p["eng_dir"]  / s["eng_mp3"]
    if not nufi_path.exists() or not eng_path.exists():
        print(f"⚠ audio missing {s['id']}")
        return

    eng  = AudioFileClip(str(eng_path))
    nufi = AudioFileClip(str(nufi_path))

    mode_l = mode.lower().strip()
    if mode_l == "lecture":
        # English → (gap) → Local
        tE        = eng.duration
        GAP       = PAUSE_BTWN_REPEAT
        st_local  = tE + GAP
        total     = st_local + nufi.duration + FINAL_PAUSE
        audio = CompositeAudioClip([eng.set_start(0), nufi.set_start(st_local)])
    else:
        # Homework: Local ×3 (with gaps) → English
        d   = nufi.duration
        t2  = d + PAUSE_BTWN_REPEAT
        t3  = t2 + d + PAUSE_BTWN_REPEAT
        stE = t3 + d + PAUSE_BTWN_REPEAT
        total = stE + eng.duration + FINAL_PAUSE
        audio = CompositeAudioClip([
            nufi.set_start(0),
            nufi.set_start(t2),
            nufi.set_start(t3),
            eng.set_start(stE),
        ])

    fs    = font_size(s)
    clips = [ImageClip(str(s["bg"])).set_duration(total)]

    # Header row
    language_clip = TextClip(p["language"], font=str(FONT_PATH),
                             fontsize=int(fs*0.55), color="yellow", method="label")
    language_clip = language_clip.set_position((15, TOP_MARGIN_PX)).set_duration(total)
    clips.append(language_clip)

    support_clip = TextClip("Please Support Resulam", font=str(FONT_PATH),
                            fontsize=int(fs*0.5), color="yellow", method="label")
    support_clip = support_clip.set_position(("right", TOP_MARGIN_PX)).set_duration(total)
    clips.append(support_clip)

    num_clip = TextClip(str(s["id"]), font=str(FONT_PATH),
                        fontsize=int(fs*0.6), color="white", method="label")
    num_clip = num_clip.set_position((VIDEO_SIZE[0] - num_clip.w - RIGHT_MARGIN_PX,
                                      TOP_MARGIN_PX + support_clip.h + int(fs*0.15)))
    num_clip = num_clip.set_duration(total)
    clips.append(num_clip)

    first_y = TOP_MARGIN_PX + max(num_clip.h, support_clip.h) + int(fs * BANNER_GAP_RATIO)

    # Captions
    if mode_l == "lecture":
        y = first_y
        y = add_cap(clips, s["source"],  0, total, y, "white",  fs)
        y = add_cap(clips, s["english"], 0, total, y, "yellow", fs)
        _ = add_cap(clips, s["french"],  0, total, y, "white",  fs)
    else:
        # Intro + staged captions
        d   = nufi.duration
        t2  = d + PAUSE_BTWN_REPEAT
        t3  = t2 + d + PAUSE_BTWN_REPEAT

        y = first_y
        y = add_cap(clips, INTRO_LINES.get(p["language"], DEFAULT_INTRO),
                     0, t2, y, "white", int(fs*0.9))
        y = add_cap(clips, "Listen, repeat and translate", 0, t2, y, "yellow", fs)

        y = first_y
        y = add_cap(clips, s["source"],  t2, total - t2, y, "white",  fs)
        y = add_cap(clips, s["english"], t3, total - t3, y, "yellow", fs)
        _ = add_cap(clips, s["french"],  t3, total - t3, y, "white",  fs)

    # Logos
    for pos in ("left", "right"):
        clips.append(ImageClip(str(LOGO_PATH)).resize(height=100)
                     .set_position((pos, "bottom")).set_duration(total))

    # Render
    tmp_aac = f"tmp-{uuid4().hex}.m4a"
    tmp_mp4 = out.with_suffix(".tmp.mp4")
    try:
        CompositeVideoClip(clips).set_audio(audio).write_videofile(
            str(tmp_mp4), fps=FPS, codec="libx264", audio_codec="aac",
            temp_audiofile=tmp_aac, remove_temp=True,
            ffmpeg_params=["-pix_fmt","yuv420p","-profile:v","high",
                           "-level","4.1","-movflags","+faststart"],
            preset="ultrafast",
            threads=FFMPEG_THREADS_PER_JOB,
        )
        shutil.move(tmp_mp4, out)
        print(f"✅ {out.name}")
    except Exception as e:
        print(f"❌ id={s['id']} {e}")
        Path(tmp_mp4).unlink(missing_ok=True)
        Path(tmp_aac).unlink(missing_ok=True)

# ── THREAD WORKER ────────────────────────────────────────────────────────
sem = threading.Semaphore(PYTHON_JOBS)

# sents = range_sents

def render_slice(sents: List[Dict], st: int, ed: int, p: Dict, mode: str):
    with sem:
        for row in sents:
            if st <= row["id"] <= ed:
                build_video(row, p, mode)

# ── MAIN PIPELINE ────────────────────────────────────────────────────────
# lang = LANGUAGE
# mode = MODE.lower().strip()
def process_language(lang: str, mode: str):
    p        = build_paths(lang, mode)
    raw      = parse_sentences(p["sent_txt"], p["lang_lower"])
    ranges   = chapter_ranges(raw)
    bgs      = [ensure_bg(b, VIDEO_SIZE) for b in list_backgrounds(p["bg_dir"])]
    sentences= tag_bgs(raw, bgs)

    # Skip sentences already built
    sentences_to_build = [
        s for s in sentences
        if not (p["out_dir"] / f"{p['lang_lower']}_sentence_{s['id']}.mp4").exists()
    ]
    if not sentences_to_build:
        print(f"✔ All videos already built for {lang} [{mode}]")
        return

    threads=[]
    for st, ed in ranges:
        range_sents = [s for s in sentences_to_build if st <= s["id"] <= ed]
        if not range_sents:
            continue
        t = threading.Thread(target=render_slice, args=(range_sents, st, ed, p, mode))
        t.start(); threads.append(t)
    for t in threads:
        t.join()
    print(f"✔ Finished {lang} [{mode}]")

# ── ENTRY POINT ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_language(LANGUAGE, MODE)
# ======================================================================================
