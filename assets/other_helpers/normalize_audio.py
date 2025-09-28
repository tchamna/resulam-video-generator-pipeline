from pydub import AudioSegment
from pathlib import Path

TARGET_RMS_DBFS = -20.0
MAX_PEAK_DBFS = -3.0
SAMPLE_RATE_HZ = 44100
BIT_RATE_KBPS = "192k"
HEAD_SILENCE_SEC = 0.7
TAIL_SILENCE_SEC = 2.0

def normalize_audio(input_path: str, suffix: str = "_normalized") -> str:
    in_path = Path(input_path)
    out_path = in_path.with_suffix(".mp3")            # force mp3
    out_path = out_path.with_name(out_path.stem + suffix + ".mp3")
    try:
        audio = AudioSegment.from_file(in_path).set_frame_rate(SAMPLE_RATE_HZ).set_channels(1)
        rms_gain = TARGET_RMS_DBFS - audio.dBFS
        audio = audio.apply_gain(rms_gain)
        if audio.max_dBFS > MAX_PEAK_DBFS:
            peak_reduction = MAX_PEAK_DBFS - audio.max_dBFS
            audio = audio.apply_gain(peak_reduction)
            if audio.dBFS < -23:
                print(f"⚠️ After peak limiting, RMS={audio.dBFS:.2f} dBFS (may be below spec).")
        head = AudioSegment.silent(duration=HEAD_SILENCE_SEC * 1000)
        tail = AudioSegment.silent(duration=TAIL_SILENCE_SEC * 1000)
        audio = head + audio + tail
        audio.export(
            out_path,
            format="mp3",
            bitrate=BIT_RATE_KBPS,
            parameters=["-acodec", "libmp3lame", "-b:a", BIT_RATE_KBPS, "-ar", str(SAMPLE_RATE_HZ), "-ac", "1"]
        )
        print(f"✅ Saved: {out_path}")
        return str(out_path)
    except Exception as e:
        print(f"❌ Error with {input_path}: {e}")
        return ""

# ---- Process multiple files ----
files = [
    r"C:\Users\tcham\Downloads\IntroDualaElevenLabWhisp.mp3",
    r"C:\Users\tcham\Downloads\EndCredit_DualaElevenLab.mp3",
    r"C:\Users\tcham\OneDrive\Documents\Camtasia\Ads Phrasebooks Videos duala Resulam.m4a"
]

for f in files:
    normalize_audio(f)
