#!/usr/bin/env python3
"""Simple dependency checker for this repo.
Run: python tools/check_deps.py
"""
import shutil
import sys
import importlib

def check_cmd(cmd):
    path = shutil.which(cmd)
    print(f"{cmd}:", path if path else "MISSING")
    return bool(path)

def check_package(pkg):
    try:
        importlib.import_module(pkg)
        print(f"{pkg}: OK")
        return True
    except Exception:
        print(f"{pkg}: MISSING")
        return False

def main():
    print("Checking CLI tools...")
    ok_ffmpeg = check_cmd('ffmpeg')
    ok_magick = check_cmd('magick')

    print('\nChecking Python packages (moviepy, pillow, pydub)')
    ok_mp = check_package('moviepy')
    ok_pil = check_package('PIL')
    ok_pydub = check_package('pydub')

    print('\nSummary:')
    print(' FFmpeg:', 'OK' if ok_ffmpeg else 'MISSING')
    print(' ImageMagick:', 'OK' if ok_magick else 'MISSING')
    print(' Python packages:', 'OK' if (ok_mp and ok_pil and ok_pydub) else 'MISSING')

    if not ok_magick:
        print('\nTip: run tools/configure_imagemagick.ps1 after installing ImageMagick to set user PATH and IMAGEMAGICK_BINARY.')

    sys.exit(0 if (ok_ffmpeg and ok_mp and ok_pil and ok_pydub) else 2)

if __name__ == '__main__':
    main()
