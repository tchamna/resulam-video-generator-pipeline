# Multi-Language Video Generator

A python script to generate educational language-learning videos with multi-language text overlays, background images, audio, and custom branding—optimized for African languages and phrasebooks.

## Features

- **Automatic video generation** from sentence lists and audio files.
- **Per-chapter background image switching**.
- **Multi-language support** (e.g., Nufi, Yoruba, Swahili, etc.).
- **Custom branding** (logo, “Please Support Resulam” banner).
- **Parallel rendering** using all CPU cores.
- **High-quality text overlays** with flexible positioning and font sizing.
- **Custom intro/headers/footers** for each video.

---

## Compatibility & Installation

⚠️ **IMPORTANT: This project does _not_ work with the latest versions of MoviePy or Pillow!**

You **must** use the specific versions below.  
Newer versions (e.g., MoviePy 2.x or Pillow 10+) will cause errors due to API changes and deprecations (notably `Image.ANTIALIAS`).

### Required Versions

```txt
moviepy==1.0.3
Pillow==9.5.0
