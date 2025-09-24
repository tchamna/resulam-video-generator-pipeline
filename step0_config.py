# step0_config.py
from pathlib import Path
import os

##############################################################
# GLOBAL SETTINGS: 
# Set Language, Mode, Env
# Use Private Assets or Normal Assets
##############################################################
LANGUAGE = "Duala"         # e.g., "Duala", "Nufi", "Yoruba"

MODE = "homework"          # "lecture" or "homework"
ENV = "test"         # "production" or "test"

# MODE = "lecture"          # "lecture" or "homework"
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
INNER_PAUSE_DURATION = 3
INNER_PAUSE_DURATION_HW = 5 # for homework mode, longer pause
# Silence duration (in seconds) to add at the begenning and at the end of each audio file.
TRAILING_PAUSE_DURATION = 1
# Number of times to repeat local audio segments (1 = no repetition)
REPEAT_LOCAL_AUDIO = 1
# Whether to add padding (silence) to make all audio segments the same length
FLAG_PAD = True

SHUFFLE_HOMEWORK = True   # or False

SHUFFLE_SEED = None             # or an int like 1234 for reproducible shuffle


# Filtering (None = process all)
# Option 1: Filter by a range of sentence numbers.
# If both sentence and chapter ranges are set, 
# the sentence range will be used.
FILTER_SENTENCE_START = 1622
FILTER_SENTENCE_END   = 1624

FILTER_SENTENCE_START = None
FILTER_SENTENCE_END   = None

# Option 2: Filter by a range of chapter numbers.
# FILTER_CHAPTER_START = 1
# FILTER_CHAPTER_END   = 3
FILTER_CHAPTER_START = None
FILTER_CHAPTER_END   = None 


# Step 2: Video Production settings

INTRO_MESSAGES = {
    "Nufi": "Yū' Mfʉ́ɑ́'sí, Mfāhngə́ə́:",
    "Swahili": "Sikiliza, rudia na tafsiri:",
    "Yoruba": "Tẹ́tí gbọ́, tunsọ, ṣe ògbùfọ̀:",
    "Duala": "Seŋgâ, Timbísɛ́lɛ̂ na Túkwâ:",
}

DEFAULT_INTRO = "Listen, repeat and translate:"

# Optional path overrides
FONT_PATH = "Fonts/arialbd.ttf"
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
MAX_WORKERS = 2
FFMPEG_THREADS = 2

# Step 3: Combining Videos settings
FPS       = 24
FFMPEG_THREADS_PER_JOB = 4

CHUNK_SIZE = 10 # Number of videos per chunk

# Control which chapters to process
SINGLE_CHAPTER = None   # e.g., 12 → only Chapter 12
START_CHAPTER  = 1     # default = 1
END_CHAPTER    = 32     # None = until last chapter

# Excluded sentence IDs
EXCLUDED_SENTENCES = {
    1476,1477,1478,1479,1480,1481,1482,1483,
    1590,1607,1610,1614,1616,1621,1628,1629,
    1646,1647,1762,1763,2010
}

