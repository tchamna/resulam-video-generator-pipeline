import os
import shutil
from natsort import natsorted
from pathlib import Path
from pydub import AudioSegment, silence
import glob
import json, subprocess
import numpy as np, soundfile as sf
import pyloudnorm as pyln 
import re

from pydub.silence import detect_nonsilent

# ─── Configuration ─────────────────────────────────────────────────────
BASE_DIR = Path(r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production")
local_language = 'duala'
silence_threshold = 1.5
silence_padding_duration = 3
trailing_slience_duration = 1
repeat_local_audio = 1
flag_pad = True
local_language_title = local_language.title()
test_or_production = "test"

# Global Silent Segments
trailing_slience = AudioSegment.silent(duration=2000)

# ─── Chapter Ranges ─────────────────────────────────────────────────────
chapter_ranges = [
    (1, 173, "Chap1"), (174, 240, "Chap2"), (241, 258, "Chap3"), (259, 407, "Chap4"),
    (408, 543, "Chap5"), (544, 568, "Chap6"), (569, 703, "Chap7"), (704, 788, "Chap8"),
    (789, 930, "Chap9"), (931, 991, "Chap10"), (992, 1011, "Chap11"), (1012, 1036, "Chap12"),
    (1037, 1074, "Chap13"), (1075, 1104, "Chap14"), (1105, 1125, "Chap15"), (1126, 1152, "Chap16"),
    (1153, 1195, "Chap17"), (1196, 1218, "Chap18"), (1219, 1248, "Chap19"), (1249, 1279, "Chap20"),
    (1280, 1303, "Chap21"), (1304, 1366, "Chap22"), (1367, 1407, "Chap23"), (1408, 1471, "Chap24"),
    (1472, 1500, "Chap25"), (1501, 1569, "Chap26"), (1570, 1650, "Chap27"), (1651, 1717, "Chap28"),
    (1718, 1947, "Chap29"), (1948, 1964, "Chap30"), (1965, 1999, "Chap31"), (2000, 2044, "Chap32")
]

Chapters_begining = [start for start, _, _ in chapter_ranges]
Chapters_ending = [end for _, end, _ in chapter_ranges]
Chapters_begining_extended = sorted(Chapters_begining + [i + 1 for i in Chapters_begining])

# ─── Path Setup ─────────────────────────────────────────────────────────
def get_nth_parent_directory(path: Path, n: int) -> Path:
    for _ in range(n):
        path = path.parent
    return path

script_dir = BASE_DIR
phrasebook_dir_name = f"{local_language_title}Only" if test_or_production == "production" else f"{local_language_title}OnlyTest"
local_language_dir = (script_dir / "Languages" / f"{local_language_title}Phrasebook" / phrasebook_dir_name).resolve()
local_audio_path = local_language_dir
main_dir = script_dir
eng_audio_path = main_dir / "EnglishOnly"
print("Starting audio directory setup...")

# ─── Subdirectory Check ─────────────────────────────────────────────────
def check_subdirectories(directory: Path) -> list:
    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if subdirs:
        print("Subdirectories found:", [str(d.name) for d in subdirs])
    else:
        print("No subdirectories found.")
    return subdirs

# ─── Audio Extraction ───────────────────────────────────────────────────
def extract_audios_and_move_original(base_path: Path, ext: str = ".mp3"):
    audio_files = []
    original_dir = base_path / "original_audios"
    os.makedirs(original_dir, exist_ok=True)
    for root, _, files in os.walk(base_path):
        root_path = Path(root)
        if root_path == original_dir or original_dir in root_path.parents:
            continue
        for f in files:
            if f.endswith(ext):
                src = root_path / f
                dest = base_path / f
                shutil.copy(src, dest)
                audio_files.append(dest)
    for subdir in check_subdirectories(base_path):
        if subdir.name != "original_audios":
            shutil.move(str(subdir), original_dir / subdir.name)
    print(f"Extracted {len(audio_files)} audio files to {base_path} and moved subdirectories to 'original_audios'")
    return natsorted(audio_files)

# ─── Helper: Get Subdirectories ─────────────────────────────────────────
def check_subdirectories(path: Path):
    return [p for p in path.iterdir() if p.is_dir()]

# ─── Audio Extraction ───────────────────────────────────────────────────
def extract_audios_and_move_original(base_path: Path, ext: str = ".mp3"):
    audio_files = []
    original_dir = base_path / "original_audios"
    os.makedirs(original_dir, exist_ok=True)

    # To move folders after walk completes
    subdirs_to_move = []

    for root, dirs, files in os.walk(base_path):
        root_path = Path(root)

        # Skip original_audios folder and its children
        if root_path == original_dir or original_dir in root_path.parents:
            continue

        # Flag to track .mp3 presence
        has_mp3 = False

        for f in files:
            if f.endswith(ext):
                has_mp3 = True
                src = root_path / f
                dest = base_path / f
                try:
                    if src.resolve() != dest.resolve():
                        shutil.copy(src, dest)
                        audio_files.append(dest)
                    else:
                        print(f"⚠️ Skipped copying '{src.name}' (source and destination are the same).")
                except Exception as e:
                    print(f"❌ Error copying {src.name}: {e}")

        if not has_mp3:
            print(f"⚠️ Skipped '{root_path.name}': No *{ext} files found.")

        # Save top-level subdirectories (excluding base path itself)
        if root_path != base_path and root_path.parent == base_path:
            subdirs_to_move.append(root_path)

    # ─── Move subdirectories after processing ─────────────────────────────
    for subdir in subdirs_to_move:
        if subdir.name != "original_audios":
            try:
                shutil.move(str(subdir), original_dir / subdir.name)
            except Exception as e:
                print(f"❌ Error moving folder '{subdir}': {e}")

    print(f"\n✅ Extracted {len(audio_files)} audio files to {base_path}")
    print(f"📁 Moved {len(subdirs_to_move)} subdirectories to '{original_dir.name}'")
    return natsorted(audio_files)

# ─── Silence Detection and Splitting ────────────────────────────────────
def split_audio_on_silence(audio_path: Path, min_silence_len=1500, silence_thresh=-40):
    audio = AudioSegment.from_file(audio_path)
    chunks = silence.split_on_silence(audio,
                                       min_silence_len=min_silence_len,
                                       silence_thresh=silence_thresh,
                                       keep_silence=0)  # Remove all silence
    return chunks
# ─── File Renaming ──────────────────────────────────────────────────────

# directory = local_audio_path
def files_rename(directory, prefix="", suffix="", replace="", by="", 
                 remove_first=0, remove_last=0,
                 lower_all=False, upper_all=True,
                 extensions=(".mp3", ".wav")):
    if isinstance(extensions, str):
        extensions = tuple(ext.strip() for ext in extensions.split(','))
    if lower_all and upper_all:
        print("Both lower_all and upper_all are set to True. Defaulting to lowercase.")
        upper_all = False
    for filename in os.listdir(directory):
        if not filename.startswith(local_language):
            original_filename = filename
            if lower_all:
                filename = filename.lower()
            elif upper_all:
                filename = filename.upper()
            try:
                new_name = filename.replace(replace, by)
                new_name = new_name[remove_first:]
                new_name = new_name[:len(new_name) - remove_last]
                new_name = prefix + new_name + suffix
            except:
                new_name = filename
            if extensions is None or (isinstance(extensions, tuple) and original_filename.lower().endswith(extensions)):
                old_file = os.path.join(directory, original_filename)
                new_file = os.path.join(directory, new_name)
                try:
                    os.rename(old_file, new_file)
                    print(f"Renamed '{original_filename}' to '{new_name}'")
                except:
                    pass
            else:
                print(f"Skipped '{original_filename}': Does not match specified extensions")


def get_audio(audio_path, 
              ext=["*.mp3", "*.wav", "*.ogg", "*.flac"], 
              check_subfolders=False):
    audio_files = []
    if check_subfolders:
        for extension in ext:
            audio_files.extend(glob.glob(os.path.join(audio_path, "**", extension), recursive=True))
    else:
        for extension in ext:
            audio_files.extend(glob.glob(os.path.join(audio_path, extension), recursive=False))
            
    audio_base_names = [os.path.basename(i) for i in audio_files]
    return natsorted(audio_files, key=lambda x: x.lower()), natsorted(audio_base_names, key=lambda x: x.lower())


def batch_process_mp3s(ref_english: str,
                       input_folder: str,
                       output_folder: str,
                       *,
                       silence_noise_db: float = -32.0,
                       silence_min_dur: float = 0.22,
                       edge_guard_ms: int = 80
                       ) -> None:

    def ffprobe_stream(path: str) -> dict:
        out = subprocess.check_output([
            "ffprobe","-v","error","-select_streams","a:0",
            "-show_entries","stream=channels,sample_rate,bit_rate",
            "-of","json", path
        ], text=True)
        streams = json.loads(out).get("streams", [])
        return streams[0] if streams else {}

    def nearest_mp3_kbps(bit_rate_bps, default=192) -> str:
        try:
            kbps = int(round(int(bit_rate_bps)/1000.0))
            common = [96,112,128,160,192,224,256,320]
            return f"{min(common, key=lambda c: abs(c-kbps))}k"
        except Exception:
            return f"{default}k"

    def integrated_lufs(path: str, fallback=-23.0) -> float:
        try:
            data, rate = sf.read(path, always_2d=False)
            if not np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32) / np.iinfo(data.dtype).max
            I = pyln.Meter(rate).integrated_loudness(data)
            return round(float(I), 1)
        except Exception:
            return float(fallback)

    def file_duration_sec(path: str) -> float:
        info = sf.info(path)
        return float(info.frames) / float(info.samplerate)

    def detect_silences(path: str, noise_db: float, min_dur: float):
        """
        Returns list of (start, end) silence intervals in seconds using FFmpeg silencedetect.
        """
        cmd = [
            "ffmpeg","-hide_banner","-nostats","-i", path,
            "-af", f"silencedetect=noise={noise_db}dB:d={min_dur}",
            "-f","null","-"
        ]
        run = subprocess.run(cmd, text=True, capture_output=True)
        silences = []
        cur_start = None
        for line in run.stderr.splitlines():
            line = line.strip()
            if "silence_start:" in line:
                try:
                    cur_start = float(line.split("silence_start:")[1].strip())
                except Exception:
                    cur_start = None
            elif "silence_end:" in line and cur_start is not None:
                try:
                    seg = line.split("silence_end:")[1].strip()
                    end = float(seg.split("|")[0].strip())
                    silences.append((cur_start, end))
                except Exception:
                    pass
                cur_start = None
        if cur_start is not None:
            silences.append((cur_start, file_duration_sec(path)))
        return silences

    ref = ffprobe_stream(ref_english)
    ENG_SR = int(ref.get("sample_rate", 44100))
    ENG_CH = int(ref.get("channels", 2))
    ENG_BR = nearest_mp3_kbps(ref.get("bit_rate", None))
    ENG_I = integrated_lufs(ref_english, fallback=-23.0)

    in_dir, out_dir = Path(input_folder), Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = natsorted(in_dir.glob("*.mp3"), key=lambda p: p.name)
    guard = edge_guard_ms / 1000.0

    for src in files:
        dst = out_dir / src.name

        silences = detect_silences(str(src), silence_noise_db, silence_min_dur)
        safe_silences = []
        for s, e in silences:
            s2, e2 = s + guard, e - guard
            if e2 > s2:
                safe_silences.append((s2, e2))

        af_parts = [
            "highpass=f=80", "lowpass=f=9000",
            "afftdn=nr=12:nt=w:om=o",
            "agate=threshold=-35dB:ratio=2:attack=5:release=50",
        ]

        for (t0, t1) in safe_silences:
            af_parts.append(f"volume=enable='between(t,{t0:.3f},{t1:.3f})':volume=0")

        af_parts += [
            "acompressor=threshold=-22dB:ratio=3:attack=8:release=120:knee=3:makeup=3",
            "dynaudnorm=f=150:g=10:n=1:p=0.7:m=5",
            f"loudnorm=I={ENG_I}:TP=-1.5:LRA=11:print_format=summary",
            "alimiter=limit=-1.5dB:level=true"
        ]
        af_chain = ",".join(af_parts)

        cmd = [
            "ffmpeg","-y","-i", str(src),
            "-vn",
            "-af", af_chain,
            "-ar", str(ENG_SR),
            "-ac", str(ENG_CH),
            "-b:a", ENG_BR,
            str(dst)
        ]
        try:
            subprocess.run(cmd, check=True, text=True, capture_output=True)
            if safe_silences:
                spans = ", ".join([f"{t0:.2f}-{t1:.2f}s" for t0, t1 in safe_silences])
                print(f"✅ {src.name} → {dst.name} (muted: {spans})")
            else:
                print(f"✅ {src.name} → {dst.name} (no silence > {silence_min_dur*1000:.0f} ms)")
        except subprocess.CalledProcessError as e:
            print(f"❌ {src.name} failed\n{e.stderr[:300]}")


def get_digits_from_string(s: str):
    m = re.search(r"\d+", s)
    return int(m.group()) if m else None

def trim_silence_ends(seg: AudioSegment, silence_thresh: int = -50, chunk_ms: int = 10) -> AudioSegment:
    spans = silence.detect_nonsilent(seg, min_silence_len=chunk_ms, silence_thresh=silence_thresh)
    if not spans:
        return AudioSegment.silent(duration=0)
    start, end = spans[0][0], spans[-1][1]
    return seg[start:end]

def normalize(seg: AudioSegment, target_dbfs: float = -20.0) -> AudioSegment:
    change = target_dbfs - seg.dBFS if seg.dBFS != float("-inf") else 0
    return seg.apply_gain(change)


def pad_and_join_chunks(seg: AudioSegment, apply_padding: bool = True) -> tuple[AudioSegment, int]:
    """
    Split audio into chunks based on silence, trim each chunk, pad in between,
    and add exactly 2s of silence at the beginning and `trailing_slience_duration` at the end.
    """

    def trim_leading_trailing_silence(audio: AudioSegment, silence_thresh: int = -50) -> AudioSegment:
        nonsilent_ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=silence_thresh)
        if not nonsilent_ranges:
            return AudioSegment.silent(duration=0)
        start = nonsilent_ranges[0][0]
        end = nonsilent_ranges[-1][1]
        return audio[start:end]

    chunks = silence.split_on_silence(
        seg,
        min_silence_len=int(silence_threshold * 1000),
        silence_thresh=-50,
        keep_silence=False
    )

    head_silence = AudioSegment.silent(duration=trailing_slience_duration * 1000)  # Always 2s
    tail_silence = AudioSegment.silent(duration=trailing_slience_duration * 1000)

    if not apply_padding or len(chunks) == 0:
        cleaned = trim_leading_trailing_silence(seg)
        final_audio = head_silence + cleaned + tail_silence
        return normalize(final_audio), 0

    if len(chunks) > 5:
        cleaned = trim_leading_trailing_silence(seg)
        final_audio = head_silence + cleaned + tail_silence
        return normalize(final_audio), 1

    padding_silence = AudioSegment.silent(duration=int(silence_padding_duration * 1000))
    segments_to_join = [head_silence]

    for i, c in enumerate(chunks):
        cleaned_chunk = trim_leading_trailing_silence(c)
        segments_to_join.append(cleaned_chunk)
        if i < len(chunks) - 1:
            segments_to_join.append(padding_silence)

    segments_to_join.append(tail_silence)

    final_audio = sum(segments_to_join, AudioSegment.silent(duration=0))
    final_audio = final_audio * max(1, repeat_local_audio)
    return normalize(final_audio), len(chunks)

# audio_dir = normalized_audio_path
def export_padded_audios(audio_dir: Path, out_dir: Path | None = None):
    files_full, _ = get_audio(audio_dir)
    bad = []
    
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        
    for src in files_full:
        try:
            src_path = Path(src)
            name = src_path.name
            num = get_digits_from_string(name)

            seg = AudioSegment.from_file(src_path)
            if num is None:
                final, _ = normalize(seg), 0
            else:
                final, _ = pad_and_join_chunks(seg, apply_padding=bool(flag_pad))

            target_dir = Path(out_dir) if out_dir else src_path.parent
            dst = target_dir / f"{src_path.stem}_padded.mp3"
            final.export(dst, bitrate="192k", format="mp3")
            print(f"✅ {name} → {dst.name}")
        except Exception as e:
            print(f"Error processing {src_path.name}: {e}")
            bad.append(Path(src).name)
            with open(Path(audio_dir) / "Bad_audios.txt", "a", encoding="utf-8") as fh:
                fh.write(f"{Path(src).name}\n")
    if bad:
        print(f"⚠ Skipped {len(bad)} files; listed in Bad_audios.txt")


# Helper: Remove silence
def remove_trailing_silence(audio_segment, silence_thresh=-50, chunk_size=10):
    nonsilent_chunks = detect_nonsilent(audio_segment, min_silence_len=chunk_size, silence_thresh=silence_thresh)
    if nonsilent_chunks:
        start_time = nonsilent_chunks[0][0]
        end_time = nonsilent_chunks[-1][1]
        return audio_segment[start_time:end_time]
    return AudioSegment.silent(duration=0)


# ────────────────────────────────────────────────────────────────
# Helper: Remove trailing silence (optional, adjust if needed)
def remove_trailing_silence(audio: AudioSegment, silence_thresh=-40.0, chunk_size=10):
    trim_ms = 0
    while trim_ms < len(audio) and audio[-trim_ms - chunk_size: -trim_ms or None].dBFS < silence_thresh:
        trim_ms += chunk_size
    return audio[:len(audio) - trim_ms]

# ────────────────────────────────────────────────────────────────
# Helper: Get audio file list


def get_audio(directory, ext=["*.mp3", "*.wav"], check_subfolders=False):
    directory = Path(directory)
    files = []
    for e in ext:
        pattern = directory / ("**" if check_subfolders else "") / e
        files += glob.glob(str(pattern), recursive=check_subfolders)

    files = natsorted([Path(f) for f in files], key=lambda x: str(x).lower())
    return files, [f.name for f in files]

# ────────────────────────────────────────────────────────────────
# Helper: Create map from _001 to file path
def create_audio_map(file_list):
    audio_map = {}
    for file in file_list:
        filename = file.name  # file is a Path object
        match = re.search(r'_(\d+)', filename)
        if match:
            audio_number = int(match.group(1))
            audio_map[audio_number] = file
        else:
            print(f"⚠️ Could not extract number from: {filename}")
    return audio_map


def merge_bilingual_audio(eng_audio_dir: Path, local_audio_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    eng_files, _ = get_audio(eng_audio_dir)
    local_files, _ = get_audio(local_audio_dir)

    eng_map = create_audio_map(eng_files)
    local_map = create_audio_map(local_files)

    for number in sorted(local_map):
        local_file = local_map[number]
        eng_file = eng_map.get(number)

        try:
            local_audio = remove_trailing_silence(AudioSegment.from_file(local_file))

            if eng_file:
                eng_audio = remove_trailing_silence(AudioSegment.from_file(eng_file))
                output_audio = Trailing_silent + eng_audio + Inside_silent + local_audio + Trailing_silent
                output_name = f"{eng_file.stem.split('_')[0]}_{local_file.name}"
            else:
                output_audio = local_audio
                output_name = f"no_english_{local_file.name}"

            output_path = out_dir / output_name
            output_audio.export(output_path, format="mp3", bitrate="192k")
            print(f"✅ Exported: {output_name}")

        except Exception as e:
            print(f"❌ Error processing file #{number}: {e}")


def get_chapter(chapter_ranges, page_number):
    for start, end, chapter in chapter_ranges:
        if start <= page_number <= end:
            return chapter
    return None

def get_digits_numbers_from_string(s):
    number = int(re.search(r'\d+', s).group())
    return number


# ─── Helper Functions ───────────────────────────────────────────────────
def get_chapter(chapter_ranges, page_number):
    for start, end, chapter in chapter_ranges:
        if start <= page_number <= end:
            return chapter
    return None

def get_digits_numbers_from_string(s):
    match = re.search(r'\d+', s)
    return int(match.group()) if match else None

#####################################################
# END FUNCTIONS 
#####################################################

# ──────────────── EXTRACT FILES FROM SUBDIR ────────────────────
subdirs = check_subdirectories(local_audio_path)
if subdirs:
    try:
        audio_files = extract_audios_and_move_original(local_audio_path)
        # local_audio_path = local_audio_path
    except Exception as e:
        print(f"Error extracting audios: {e}")
        audio_files = []
        # local_audio_path = local_audio_path

# ──────────────────── RENAME FILES ────────────────────────────────

prefix = f"{local_language.lower()}_phrasebook_"
files_rename(local_audio_path, 
             prefix=prefix, suffix="", 
             replace="", by="", 
             remove_first=0, remove_last=0, 
             lower_all=True, upper_all=False,
             extensions=(".mp3", ".wav"))

# ─────────────── NORMALIZE AUDIO ─────────────────────────

audio_files_plus_ext, audio_files = get_audio(local_audio_path)

REFERENCE_AUDIO = os.path.join(BASE_DIR,"Languages","reference_audio.mp3")

normalized_audio_path = local_audio_path / "gen1_normalized"

batch_process_mp3s(
    ref_english=REFERENCE_AUDIO,
    input_folder=local_audio_path,
    output_folder=normalized_audio_path,
    silence_noise_db=-32.0,
    silence_min_dur=0.22,
    edge_guard_ms=80
)
# ─────────────── PAD NORMALIZED AUDIO ─────────────────────────
normalized_padded_path = local_audio_path / "gen2_normalized_padded"

# Run the second audio padding process with the new output directory
export_padded_audios(normalized_audio_path, out_dir=normalized_padded_path)

# ──────────────  BILINGUAL AUDIO PRODUCTION ───────────────────────────────

Trailing_silent = AudioSegment.silent(duration=1000)
Inside_silent = AudioSegment.silent(duration=3000)

bilingual_output_path = local_language_dir / "gen3_bilingual_sentences"
os.makedirs(bilingual_output_path, exist_ok=True)

merge_bilingual_audio(eng_audio_path, normalized_padded_path, bilingual_output_path)

# ──────────────  PER CHAPTER AUDIO PRODUCTION ───────────────────────────────
###########################################################
# Combine Sentences in Chapters
###########################################################

# Inputs
# bilingual_output_path = local_language_dir / "gen3_bilingual_sentences"
combined_chapters_audio_folder = bilingual_output_path / "bilingual_sentences_chapters"
os.makedirs(combined_chapters_audio_folder, exist_ok=True)

# Load audio files
audio_files_plus_ext, audio_files = get_audio(
    bilingual_output_path,
    ext=["*.mp3", "*.wav", "*.ogg", "*.flac"],
    check_subfolders=False
)

# Sort
audio_files_plus_ext = natsorted(audio_files_plus_ext, key=lambda x: str(x).lower())

# ─── Chapter Combination Logic ──────────────────────────────────────────
song = None
previous_chapter = None

for idx, audio in enumerate(audio_files_plus_ext):
    base_audio = os.path.basename(audio)

    # Skip invalid files
    if "mistake" in str(audio).lower() or base_audio[0].isdigit():
        continue

    audio_num = get_digits_numbers_from_string(base_audio)
    chapter_num = get_chapter(chapter_ranges, audio_num)

    if not chapter_num:
        print(f"⚠️ Could not determine chapter for {base_audio} (num={audio_num})")
        continue

    print(f"→ Processing {base_audio} → {chapter_num}")

    song_i = AudioSegment.from_file(audio)

    # First iteration
    if song is None:
        song = song_i
        previous_chapter = chapter_num
        continue

    # If chapter changed, export previous
    if chapter_num != previous_chapter:
        output_path = combined_chapters_audio_folder / f"phrasebook_{local_language}_{previous_chapter}.mp3"
        song.export(output_path, format="mp3", bitrate="192k")
        print(f"✅ Exported {output_path.name}")

        # Start new song
        song = song_i
        previous_chapter = chapter_num
    else:
        song += song_i

# Final export
if song and previous_chapter:
    output_path = combined_chapters_audio_folder / f"phrasebook_{local_language}_{previous_chapter}.mp3"
    if not output_path.exists():  # <-- Only export if not already written
        song.export(output_path, format="mp3", bitrate="192k")
        print(f"✅ Exported {output_path.name}")
    else:
        print(f"⚠️ Skipped re-exporting {output_path.name} (already written)")

