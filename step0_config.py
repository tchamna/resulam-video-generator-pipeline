# step0_config.py
from pathlib import Path
import os

##############################################################
# GLOBAL SETTINGS: 
# Set Language, Mode, Env
# Use Private Assets or Normal Assets
##############################################################
LANGUAGE = "Duala"         # e.g., "Duala", "Nufi", "Yoruba"
# LANGUAGE = "Bamoun"         # e.g., "Duala", "Nufi", "Yoruba"

MODE = "homework"          # "lecture" or "homework"
ENV = "production"         # "production" or "test"

MODE = "lecture"          # "lecture" or "homework"
# ENV = "test"         # "production" or "test"

# No need to change the lines below, unless you know what you're doing

# ─── Asset Source ──────────────────────────────────
# Use env var override if set, otherwise default here

BASE_DIR        = Path(os.getcwd())

if "USE_PRIVATE_ASSETS" not in os.environ:
    os.environ["USE_PRIVATE_ASSETS"] = "1"

USE_PRIVATE_ASSETS = os.getenv("USE_PRIVATE_ASSETS") == "1"



##############################################################
# ─── Step-specific settings ──────────────────────────────
##############################################################

# No need to change the lines below, unless you know what you're doing

# ─── Step 1: Audio Processing ──────────────────────────────

#  Chuncks separated by silence longer than this are considered distinct. 
SILENCE_THRESH = 1.5 
# Silence duration (in seconds) to add between consecutive chuncks.
INNER_PAUSE_DURATION = 4
INNER_PAUSE_DURATION_HW = 5 # for homework mode, longer pause
# Silence duration (in seconds) to add at the begenning and at the end of each audio file.
TRAILING_PAUSE_DURATION = 1
# Number of times to repeat local audio segments (1 = no repetition)
REPEAT_LOCAL_AUDIO = 2
# Whether to add padding (silence) to make all audio segments the same length
FLAG_PAD = True

SHUFFLE_HOMEWORK = True   # or False

SHUFFLE_SEED = None             # or an int like 1234 for reproducible shuffle


# Filtering (None = process all)
# Option 1: Filter by a range of sentence numbers.
# If both sentence and chapter ranges are set, 
# the sentence range will be used.
START_SENTENCE, END_SENTENCE = 12, 12
START_SENTENCE, END_SENTENCE = None, None

# Option 2: Filter by a range of chapter numbers.
START_CHAPTER, END_CHAPTER = 1, 1
START_CHAPTER, END_CHAPTER = None, None 


# Step 2: Video Production settings

INTRO_MESSAGES = {
    "Nufi": "Yū', Mfʉ́ɑ́'sí, Mfāhngə́ə́:",
    "Swahili": "Sikiliza, rudia na tafsiri:",
    "Yoruba": "Tẹ́tí gbọ́, tunsọ, ṣe ògbùfọ̀:",
    "Duala": "Seŋgâ, Timbísɛ́lɛ̂ na Túkwâ:",
    "Bamoun": "Yú', Mbééshə́ mbi' ntóóshə́:",
}

DEFAULT_INTRO = "Listen, repeat and translate:"

# Optional path overrides
# FONT_PATH = "Fonts/arialbd.ttf"
FONT_PATH = "Fonts/CharisSIL-B.ttf"
LOGO_PATH = "resulam_logo_resurrectionLangue.png"
LOG_DIR = f"Languages/{LANGUAGE}Phrasebook/Logs"


# ─── Video Production ──────────────────────────────
VIDEO_RESOLUTION = (1920, 1080)
FRAME_RATE = 24
# TRAILING_PAUSE_DURATION = 3
DEFAULT_FONT_SIZE = 100

# ─── Combining Videos ─────────────────────────────────────
CHUNK_SIZE = 10
EXCLUDED_SENTENCES = set()

# ─── Background Music ──────────────────────────────
MUSIC_FILENAME = f"{LANGUAGE.lower()}_music_background.mp3"
MUSIC_VOLUME = 0.1

# ─── Parallelism ───────────────────────────────────
USE_PARALLEL = True
# MAX_WORKERS = 4
FFMPEG_THREADS = 2

# Auto-tune parallelism defaults when not provided via environment or explicit config
try:
    _CPU_COUNT = os.cpu_count() or 1
except Exception:
    _CPU_COUNT = 1

MAX_WORKERS = _CPU_COUNT // 2

# Allow env override first
if "MAX_WORKERS" in os.environ:
    try:
        MAX_WORKERS = max(1, int(os.environ["MAX_WORKERS"]))
    except Exception:
        pass
if "FFMPEG_THREADS" in os.environ:
    try:
        FFMPEG_THREADS = max(1, int(os.environ["FFMPEG_THREADS"]))
    except Exception:
        pass

# If either value is missing or non-sensical, compute sensible defaults
if not isinstance(MAX_WORKERS, int) or MAX_WORKERS <= 0:
    # Aim to keep a few workers so each has threads available
    MAX_WORKERS = max(1, _CPU_COUNT // max(1, FFMPEG_THREADS))

if not isinstance(FFMPEG_THREADS, int) or FFMPEG_THREADS <= 0:
    # Give each job at least one thread; balance threads/workers
    FFMPEG_THREADS = max(1, _CPU_COUNT // max(1, MAX_WORKERS))

# Clamp values to avoid oversubscription
if FFMPEG_THREADS > _CPU_COUNT:
    FFMPEG_THREADS = max(1, _CPU_COUNT // MAX_WORKERS)
if MAX_WORKERS > _CPU_COUNT:
    MAX_WORKERS = _CPU_COUNT

# Step 3: Combining Videos settings
FPS       = 24
FFMPEG_THREADS_PER_JOB = 4

CHUNK_SIZE = 10 # Number of videos per chunk

# Control which chapters to process
SINGLE_CHAPTER = None   # e.g., 12 → only Chapter 12

# Excluded sentence IDs
EXCLUDED_SENTENCES = {
    1476,1477,1478,1479,1480,1481,1482,1483,
    1590,1607,1610,1614,1616,1621,1628,1629,
    1646,1647,1762,1763,2010
}

# Optional neural-network noise reduction (external tool like RNNoise)
# When True, pipeline will attempt to use RNNoise via command-line if available.
USE_NN_NOISE_REDUCTION = False
# Path to RNNoise executable or wrapper (if installed). If blank, the pipeline will
# try to call `rnnoise` on PATH when USE_NN_NOISE_REDUCTION is True.
RNNOISE_BINARY_PATH = os.getenv("RNNOISE_BINARY_PATH", "")

