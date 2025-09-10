from pathlib import Path
from pydub import AudioSegment, silence

# ── CONFIG ─────────────────────────────────────────────────────────────────
IN_DIR  = Path(r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\Languages\DualaPhrasebook\DualaOnly\cleaned")
OUT_DIR = IN_DIR.parent / "silence_fixed"
OUT_DIR.mkdir(exist_ok=True)

SILENCE_THRESH_DBFS = -40     # adjust if needed
MIN_SIL_CHUNK_MS    = 100     # granularity for silence detection

LEAD_TAIL_MS        = 1000    # exactly 1s at start and end
INTERNAL_TARGET_MS  = 3000    # exactly 3s for long internal silences
INTERNAL_MIN_MS     = 2000    # threshold above which we clamp/extend to 3s

# ── HELPERS ────────────────────────────────────────────────────────────────
def make_silence(ms: int, like: AudioSegment) -> AudioSegment:
    return (AudioSegment
            .silent(duration=ms, frame_rate=like.frame_rate)
            .set_sample_width(like.sample_width)
            .set_channels(like.channels))

def detect_leading_silence(seg: AudioSegment, thresh_dbfs: int, step_ms: int = 10) -> int:
    trim_ms = 0
    while trim_ms < len(seg) and seg[trim_ms:trim_ms+step_ms].dBFS < thresh_dbfs:
        trim_ms += step_ms
    return trim_ms

def detect_trailing_silence(seg: AudioSegment, thresh_dbfs: int, step_ms: int = 10) -> int:
    trim_ms = 0
    while trim_ms < len(seg) and seg[-trim_ms-step_ms:len(seg)-trim_ms].dBFS < thresh_dbfs:
        trim_ms += step_ms
    return trim_ms

# ── CORE LOGIC ─────────────────────────────────────────────────────────────
def process_one(path: Path) -> None:
    audio = AudioSegment.from_file(path)

    # Find non-silent islands (to compute internal gaps)
    nonsilent = silence.detect_nonsilent(
        audio,
        min_silence_len=MIN_SIL_CHUNK_MS,
        silence_thresh=SILENCE_THRESH_DBFS
    )

    if not nonsilent:
        # Entire file is silence → just output 1s + 1s
        out = make_silence(LEAD_TAIL_MS, audio) + make_silence(LEAD_TAIL_MS, audio)
        out.export(OUT_DIR / path.name, format=path.suffix.lstrip("."))
        print(f"[OK] {path.name} (all silence)")
        return

    # Trim to exactly 1s at start/end (remove any existing edge silence first)
    lead_ms   = detect_leading_silence(audio, SILENCE_THRESH_DBFS)
    trail_ms  = detect_trailing_silence(audio, SILENCE_THRESH_DBFS)
    core      = audio[lead_ms: len(audio)-trail_ms] if trail_ms < len(audio) else audio[lead_ms:]

    # Recompute non-silent islands inside the trimmed core
    core_ns = silence.detect_nonsilent(
        core,
        min_silence_len=MIN_SIL_CHUNK_MS,
        silence_thresh=SILENCE_THRESH_DBFS
    )

    # Build the output: 1s head + [island + adjusted gap] + 1s tail
    out = make_silence(LEAD_TAIL_MS, audio)

    for i, (s, e) in enumerate(core_ns):
        # Add the non-silent chunk
        out += core[s:e]

        # Add gap to next island (internal silence)
        if i < len(core_ns) - 1:
            this_end   = e
            next_start = core_ns[i+1][0]
            gap_ms     = max(0, next_start - this_end)

            if gap_ms > INTERNAL_MIN_MS:
                gap_ms = INTERNAL_TARGET_MS  # clamp/extend to exactly 3s

            out += make_silence(gap_ms, audio)

    # Add exactly 1s trailing silence
    out += make_silence(LEAD_TAIL_MS, audio)

    # Export with original extension
    out_path = OUT_DIR / path.name
    out.export(out_path, format=path.suffix.lstrip("."))
    print(f"[OK] {path.name} → {out_path.name}")

def main():
    exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    files = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in exts and p.is_file()])
    if not files:
        print(f"No audio files found in: {IN_DIR}")
        return
    for f in files:
        try:
            process_one(f)
        except Exception as e:
            print(f"[ERR] {f.name}: {e}")

if __name__ == "__main__":
    main()
