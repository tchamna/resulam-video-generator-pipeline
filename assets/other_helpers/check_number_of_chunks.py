import os
import shutil
from pathlib import Path
from pydub import AudioSegment, silence

# ── CONFIG ──────────────────────────────────────────────
input_folder = Path(r"D:\Resulam\Videos_Production\private_assets\Languages\DualaPhrasebook\Results_Audios\gen2_normalized_padded")  # change to your folder
output_folder = Path(r"D:\Resulam\Videos_Production\private_assets\Languages\DualaPhrasebook\Results_Audios\gen2_normalized_padded\Odd_Chunks")
output_folder.mkdir(parents=True, exist_ok=True)

# Silence detection settings (tune if needed)
MIN_SILENCE_LEN = 2000   # ms (2s)
SILENCE_THRESH = -40     # dBFS (relative silence threshold)

# ── MAIN ───────────────────────────────────────────────
for file in input_folder.glob("*.mp3"):
    try:
        audio = AudioSegment.from_file(file)
        chunks = silence.split_on_silence(
            audio,
            min_silence_len=MIN_SILENCE_LEN,
            silence_thresh=SILENCE_THRESH,
            keep_silence=0
        )
        num_chunks = len(chunks)
        print(f"{file.name}: {num_chunks} chunks")

        if num_chunks % 2 != 0:  # odd number of chunks
            dest = output_folder / file.name
            shutil.copy(file, dest)
            print(f"➡️ Copied {file.name} (odd number of chunks)")

    except Exception as e:
        print(f"❌ Error processing {file.name}: {e}")
