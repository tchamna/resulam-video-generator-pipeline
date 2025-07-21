"""
multi_language_video_generator.py  • 2025-07-20 (k)
────────────────────────────────────────────────────────────────────────────
• LANGUAGE switch: "Nufi", "Yoruba", …
• One background per chapter (detects “Chapter …” lines)
• CPU-aware parallelism (FFMPEG_THREADS_PER_JOB × PYTHON_JOBS)
• Header layout
      <Language>   ……   “Please Support Resulam”
      sentence-number        (next row, right-aligned)
• Footer URL call left in place but commented out
• Logos resized to 100 px
"""

from __future__ import annotations
import os, shutil, threading
from pathlib import Path
from uuid import uuid4
from PIL import Image, ImageFile
from moviepy.editor import (
    AudioFileClip, CompositeAudioClip, CompositeVideoClip,
    ImageClip, TextClip,
)

# ── PARALLELISM SETTINGS ────────────────────────────────────────────────
FFMPEG_THREADS_PER_JOB = 4             # threads per ffmpeg encode
PYTHON_JOBS            = 5             # concurrent MoviePy jobs

logical = os.cpu_count() or 1
while FFMPEG_THREADS_PER_JOB > logical:
    FFMPEG_THREADS_PER_JOB //= 2 or 1
PYTHON_JOBS = min(PYTHON_JOBS,
                  max(1, logical // FFMPEG_THREADS_PER_JOB))

# ── BASIC PATH / FONT SETTINGS ───────────────────────────────────────────
LANGUAGE     = "Yoruba"

ROOT         = Path(r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production")
FONT_PATH    = Path(r"C:\Users\tcham\OneDrive\Documents\Workspace_Codes\dictionnaire-nufi-franc-nufi"
                    r"\app\src\main\assets\fonts\CharisSIL-B.ttf")
LOGO_PATH    = Path(r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production_Backup"
                    r"\resulam_logo_resurrectionLangue.png")

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
    "Yoruba":  "Tẹ́tí gbọ́, tunsọ, ṣe ògbùfọ̀:",
}
DEFAULT_INTRO = "Listen, repeat and translate:"
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── PATH HELPERS ─────────────────────────────────────────────────────────
def build_paths(lang: str):
    title, slug = lang, lang.lower()
    out_dir = ROOT / f"Python_Scripts_Resulam_Phrasebooks_Audio_Processing/{title}Homework"
    os.makedirs(out_dir, exist_ok=True)
    return {
        "language": title,
        "slug": slug,
        "bg_dir":   ROOT / "Backgrounds_Selected",
        "nufi_dir": ROOT / f"Languages/{title}Phrasebook/{title}Only",
        "eng_dir":  ROOT / "EnglishOnly",
        "out_dir":  out_dir,
        "sent_txt": ROOT / f"Languages/{title}Phrasebook/{slug}_english_french_phrasebook_sentences_list.txt",
    }

def list_backgrounds(dir_: Path):
    imgs = [p for ext in ("*.png","*.jpg","*.jpeg") for p in dir_.glob(ext)]
    if not imgs:
        raise RuntimeError(f"No backgrounds in {dir_}")
    return imgs

# ── SENTENCE PARSER ──────────────────────────────────────────────────────
def parse_sentences(txt: Path, slug: str):
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
                    "nufi_mp3": f"{slug}_phrasebook_{sid}.mp3",
                    "eng_mp3":  f"english_{sid}.mp3",
                })
            except ValueError:
                print(f"⚠ bad line: {ln.strip()}")
    return rows

# ── CHAPTER RANGES ───────────────────────────────────────────────────────
def chapter_ranges(sents):
    ranges, start = [], None
    for s in sorted(sents, key=lambda x: x["id"]):
        marker = s["english"].lower().startswith("chapter")
        if marker:
            if start is not None:
                ranges.append((start, prev))
            start = s["id"]                  # include marker
        prev = s["id"]
    if start is not None:
        ranges.append((start, prev))
    return ranges

# ── BACKGROUND UTILITIES ────────────────────────────────────────────────
def ensure_bg(img: Path, tgt=VIDEO_SIZE):
    with Image.open(img) as im:
        if im.size == tgt:
            return img
        even = (tgt[0]-tgt[0]%2, tgt[1]-tgt[1]%2)
        out  = img.with_stem(img.stem + "_resized")
        im.resize(even, Image.LANCZOS).save(out)
        return out

def tag_bgs(sents, bgs):
    idx, cur, out = -1, bgs[0], []
    for s in sorted(sents, key=lambda x: x["id"]):
        if s["english"].lower().startswith("chapter"):
            idx = (idx + 1) % len(bgs); cur = bgs[idx]
        d = dict(s); d["bg"] = cur; out.append(d)
    return out

# ── CAPTION HELPERS ──────────────────────────────────────────────────────
def font_size(s):
    n=len(f"{s['source']} {s['english']} {s['french']}")
    return (BASE_FS if n<50 else int(BASE_FS*0.85) if n<80
            else int(BASE_FS*0.75) if n<110 else int(BASE_FS*0.65))

def add_cap(lst, txt, start, dur, y, color, fs, wrap=True, align="center"):
    mode="caption" if wrap else "label"
    size=(VIDEO_SIZE[0]-200,None) if wrap else None
    clip=(TextClip(txt,font=str(FONT_PATH),fontsize=fs,color=color,
                   method=mode,size=size)
          .set_position((align,y)).set_start(start).set_duration(dur))
    lst.append(clip); return y+clip.h+int(fs*0.20)

# ── VIDEO BUILDER ────────────────────────────────────────────────────────
def build_video(s, p):
    out = p["out_dir"] / f"{p['slug']}_sentence_{s['id']}.mp4"
    if out.exists(): return

    nufi_path = p["nufi_dir"] / s["nufi_mp3"]
    eng_path  = p["eng_dir"]  / s["eng_mp3"]
    if not nufi_path.exists() or not eng_path.exists():
        print(f"⚠ audio missing {s['id']}")
        return

    nufi = AudioFileClip(str(nufi_path))
    eng  = AudioFileClip(str(eng_path))

    d   = nufi.duration
    t2  = d + PAUSE_BTWN_REPEAT
    t3  = t2 + d + PAUSE_BTWN_REPEAT
    stE = t3 + d + PAUSE_BTWN_REPEAT
    total = stE + eng.duration + FINAL_PAUSE

    audio = CompositeAudioClip([nufi.set_start(0), nufi.set_start(t2),
                                nufi.set_start(t3), eng.set_start(stE)])

    fs   = font_size(s)
    clips= [ImageClip(str(s["bg"])).set_duration(total)]

    # Header row
    clips.append(TextClip(p["language"], font=str(FONT_PATH),
                          fontsize=int(fs*0.55), color="yellow", method="label")
                 .set_position((15, TOP_MARGIN_PX)).set_duration(total))

    support = TextClip("Please Support Resulam", font=str(FONT_PATH),
                       fontsize=int(fs*0.5), color="yellow", method="label"
                 ).set_position(("right", TOP_MARGIN_PX)).set_duration(total)
    clips.append(support)

    num = TextClip(str(s["id"]), font=str(FONT_PATH),
                   fontsize=int(fs*0.6), color="white", method="label")
    num = num.set_position((VIDEO_SIZE[0]-num.w-RIGHT_MARGIN_PX,
                            TOP_MARGIN_PX+support.h+int(fs*0.15))
            ).set_duration(total)
    clips.append(num)

    first_y = TOP_MARGIN_PX + max(num.h, support.h) + int(fs*BANNER_GAP_RATIO)

    y = first_y
    y = add_cap(clips, INTRO_LINES.get(p["language"], DEFAULT_INTRO),
                0, t2, y, "white", int(fs*0.9))
    y = add_cap(clips, "Listen, repeat and translate", 0, t2, y, "yellow", fs)

    y = first_y
    y = add_cap(clips, s["source"],  t2, total-t2, y, "white", fs)
    y = add_cap(clips, s["english"], t3, total-t3, y, "yellow", fs)
    add_cap(clips, s["french"],      t3, total-t3, y, "white", fs)

    # Footer URL (currently disabled)
    # add_cap(clips,"www.resulam.com",0,total,VIDEO_SIZE[1]-95,
    #         "yellow",int(fs*0.5),wrap=False)

    # Logos
    for pos in ("left","right"):
        clips.append(ImageClip(str(LOGO_PATH)).resize(height=100)
                     .set_position((pos,"bottom")).set_duration(total))

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
        shutil.move(tmp_mp4, out); print(f"✅ {out.name}")
    except Exception as e:
        print(f"❌ id={s['id']} {e}"); tmp_mp4.unlink(missing_ok=True)

# ── THREAD WORKER ────────────────────────────────────────────────────────
sem = threading.Semaphore(PYTHON_JOBS)
def render_slice(sents, st, ed, p):
    with sem:
        for row in sents:
            if st <= row["id"] <= ed:
                build_video(row, p)

# ── MAIN PIPELINE ────────────────────────────────────────────────────────
def process_language(lang: str):
    p        = build_paths(lang)
    raw      = parse_sentences(p["sent_txt"], p["slug"])
    ranges   = chapter_ranges(raw)
    bgs      = [ensure_bg(b, VIDEO_SIZE) for b in list_backgrounds(p["bg_dir"])]
    sentences= tag_bgs(raw, bgs)

    threads=[]
    for st, ed in ranges:
        t = threading.Thread(target=render_slice,args=(sentences, st, ed, p))
        t.start(); threads.append(t)
    for t in threads: t.join()
    print(f"✔ Finished {lang}")

# ── ENTRY POINT ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    process_language(LANGUAGE)
