import shutil
import re
from pathlib import Path
from pydub import AudioSegment, silence
from natsort import natsorted

# ── CONFIG ──────────────────────────────────────────────
input_folder = Path(r"D:\Resulam\Videos_Production\private_assets\Languages\DualaPhrasebook\Results_Audios\gen2_normalized_padded")
output_folder = input_folder / "Odd_Chunks"
output_folder.mkdir(parents=True, exist_ok=True)

# Silence detection settings
MIN_SILENCE_LEN = 2000   # ms (2s)
SILENCE_THRESH = -40     # dBFS

# ── MAIN ───────────────────────────────────────────────
files = natsorted(list(input_folder.glob("*.mp3")), key=lambda f: f.name.lower())
odd_files = []

def extract_sentence_number(filename: str) -> int:
    """Extracts the number from filenames like duala_phrasebook_1481_padded.mp3"""
    match = re.search(r"(\d+)", filename)
    return int(match.group(1)) if match else -1

for file in files:
    sentence_num = extract_sentence_number(file.name)

    # Process only if sentence number > 1480
    if sentence_num > 1480:
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
                odd_files.append((file.name, num_chunks))
                print(f"➡️ Copied {file.name} (odd number of chunks)")

        except Exception as e:
            print(f"❌ Error processing {file.name}: {e}")

# ── SUMMARY ─────────────────────────────────────────────
print("\n====== SUMMARY ======")
print(f"Total files checked (all): {len(files)}")
print(f"Files checked (>1480): {len([f for f in files if extract_sentence_number(f.name) > 1480])}")
print(f"Odd files found: {len(odd_files)}")
for fname, chunks in odd_files:
    print(f" - {fname} → {chunks} chunks")
