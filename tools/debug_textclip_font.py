import sys
from pathlib import Path
import traceback

root = Path(__file__).resolve().parents[1]
if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from moviepy.config import get_setting
from shutil import which

print('MoviePy IMAGEMAGICK_BINARY setting:', get_setting('IMAGEMAGICK_BINARY'))
print('magick on PATH:', which('magick'))
print('convert on PATH:', which('convert'))

# Font candidates
import step0_config as cfg
FONT_PATH = cfg.FONT_PATH
print('Configured FONT_PATH:', FONT_PATH)

cands = []
try:
    p = Path(FONT_PATH)
    cands.extend([str(p), str(p.resolve()), p.name, p.stem, str(p).replace('\\','/')])
except Exception:
    cands.append(str(FONT_PATH))
cands.append(str(FONT_PATH))

from moviepy.editor import TextClip

for f in cands:
    print('\nTrying font spec:', f)
    try:
        tc = TextClip('Test 123 — Masóma!', font=f, fontsize=80, color='yellow', method='label')
        print('Success: TextClip created (w,h)=', getattr(tc,'w',None), getattr(tc,'h',None))
    except Exception as e:
        print('TextClip failed:')
        traceback.print_exc()

# If ImageMagick present, try listing fonts
from subprocess import Popen, PIPE, STDOUT
magick = which('magick') or get_setting('IMAGEMAGICK_BINARY')
if magick and magick not in ('auto-detect','unset',None):
    try:
        print('\nRunning "magick -list font | more" (first 2000 chars):')
        p = Popen([magick, '-list', 'font'], stdout=PIPE, stderr=STDOUT, universal_newlines=True)
        out, _ = p.communicate(timeout=10)
        print(out[:2000])
    except Exception as e:
        print('Failed to run magick -list font:')
        traceback.print_exc()
else:
    print('\nNo magick binary detected to list fonts.')
