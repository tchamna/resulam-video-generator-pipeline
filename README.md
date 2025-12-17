# Multi-Language Audio Processing and Video Generator


![alt text](assets/demo_images/demo_resulam_video_generation_pipeline.jpg)




This Python script is an automated video production pipeline for creating educational language-learning videos. It generates videos with synchronized multi-language text overlays, dynamic background images, and custom branding from audio files and sentence lists. Optimized for the "Guide de conversation trilingue" series, this tool streamlines the creation of engaging content for African languages.

The pipeline consists of modular, independent scripts. Each module can be executed separately, provided the necessary audio, text, and image resources are available.


![Demo of pipeline result](assets/demo_images/demo_image1.png)

## Core Features ✨

* **Automated Generation**: Produces videos directly from organized sentence lists and audio files.
* **Dynamic Backgrounds**: Switches background images for each new chapter.
* **Multi-Language Support**: Supports a wide range of African languages.
* **Custom Branding**: Includes customizable branding with a logo and a "Please Support Resulam" banner.
* **Parallel Processing**: Utilizes available CPU cores for efficient parallel video rendering.

The script was initialy designed to produce videos for the book "Guide de conversation trilingue Français-anglais-fe'efe'e: French-Fè'éfě'è-English Phrasebook" series by Shck Tchamna [https://www.amazon.com/dp/B099TNLQL2, https://www.amazon.com/dp/B09B5DSVKL]. The books have been translated into multiple African languages, including Wolof, Duala, Ewondo, Swahili, Ewe, Kikongo, Yoruba, Igbo, Hausa, Fulfulde, and others. They consist of approximately 2044 sentences across 32 chapters, organized as follows:
"1-173": "Chap1",
"174-240": "Chap2",
"241-258": "Chap3",
...
"1965-1999": "Chap31",
"2000-2044": "Chap32"

## Compatibility & Installation

⚠️ **IMPORTANT: This project does _not_ work with the latest versions of MoviePy or Pillow!**

API changes in newer releases (e.g., MoviePy 2.x, Pillow 10+) will cause critical errors. Newer versions (e.g., MoviePy 2.x or Pillow 10+) will cause errors due to API changes and deprecations (notably `Image.ANTIALIAS`).

Install the required versions directly:

```bash
moviepy==1.0.3
Pillow==9.5.0
```

For streamlined installation, use the provided requirements.txt file:
`pip install -r requirements.txt`


## FFmpeg Installation (Required)

This project relies heavily on FFmpeg for audio and video processing. You must have FFmpeg installed and available in your system PATH for the pipeline to work.

### How to install FFmpeg on Windows

1. Download FFmpeg:
	- Go to https://www.gyan.dev/ffmpeg/builds/
	- Download the latest "release full" build (for example `ffmpeg-release-full.7z`).
2. Extract the archive:
	- Use 7-Zip (https://www.7-zip.org/) or a similar extractor to unpack the `.7z` file.
	- Inside the extracted folder find the `bin` directory (it contains `ffmpeg.exe`).
3. Add FFmpeg to your system PATH:
	- Copy the full path to the `bin` folder (for example `C:\ffmpeg\bin`).
	- Open PowerShell as Administrator and run:
	  ```powershell
	  $ffmpegPath = 'C:\path\to\ffmpeg\bin'  # change this to your actual path
	  [Environment]::SetEnvironmentVariable('Path', $env:Path + ";$ffmpegPath", [EnvironmentVariableTarget]::Machine)
	  ```
	- Or add the path via Windows Settings → System → About → Advanced system settings → Environment Variables.
4. Verify installation:
	- Restart PowerShell and run:
	  ```powershell
	  ffmpeg -version
	  ```
	- You should see FFmpeg version information printed. If not, re-check the PATH you added.

Important: After updating your PATH, restart Visual Studio Code so the editor and its integrated terminals pick up the new environment.

FFmpeg is required for the audio/video processing steps in this pipeline. If `ffmpeg -version` fails, the scripts that call ffmpeg will raise a FileNotFoundError.

![FFmpeg and ImageMagick diagram](assets/demo_images/ffmpeg_and_ImageMagick.png)

# Image rendering backend (ImageMagick vs Pillow)

Historically MoviePy's `TextClip` used ImageMagick to rasterize text. That required a working ImageMagick install on PATH and could raise errors like "WinError 2" when the binary wasn't found.

This repository's video production script (`step2_video_production.py`) is configured to avoid that dependency by default: it explicitly disables ImageMagick and forces MoviePy to use Pillow / FreeType for text rendering where possible. Concretely, `step2_video_production.py` calls:

```python
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": None})
```

What this means for you:
- You do NOT need to install ImageMagick just to run `step2_video_production.py` in the typical pipeline flow — the script will prefer Pillow/FreeType and will also fall back to in-repo rasterization for some labels using the bundled Charis TTF.
- If you run other scripts that still call `TextClip` with ImageMagick-specific features, or if you explicitly override settings, ImageMagick may still be required.
- If you want to opt back into ImageMagick/TextClip, set `USE_IMAGEMAGICK=1` (optionally also set `IMAGEMAGICK_BINARY=magick`).

If you prefer to install ImageMagick system-wide (optional):

1. Install quickly with winget (recommended):
	```powershell
	winget install --id ImageMagick.ImageMagick -e
	```
	Or with Chocolatey:
	```powershell
	choco install imagemagick -y
	```
	Or download the installer from https://imagemagick.org and run it.
2. During installation, enable the option to add ImageMagick to the system PATH if prompted.
3. Verify installation in a new PowerShell session:
	```powershell
	magick -version
	```

If you install ImageMagick but MoviePy still can't find it, you can explicitly tell MoviePy which binary to use:

```python
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "magick"})
```

Important: After installing ImageMagick or changing PATH, restart Visual Studio Code so the editor and its integrated terminals pick up the new environment.

# Step 1: Audio Processing 
![Demo audio processing](assets/demo_images/audio_processing.png)

The audio processing module prepares raw audio files for video integration. This is a crucial first step in the pipeline.

**Key Features**
* **Audio Normalization**: Standardizes the loudness (LUFS) and all other audio features of all audio files to match a specified reference track.

* **Silence Padding** : Inserts a consistent duration of silence between audio chunks (e.g., between individual phrases in a sentence) to enhance clarity and pacing. This transforms audio segments like (Phrase 1) + **0.5s silence** + (Phrase 2) into (Phrase 1) + **2s silence** + (Phrase 2).

* **Bilingual Merging** : Automatically combines a foreign language audio track with a local language track.
For instance, `english_1.mp3` and `yoruba_1.mp3` would be merged to form `english_1_yoruba_1.mp3`. Likewise, `english_203.mp3` and `yoruba_203.mp3` would be merged to form `english_203_yoruba_203.mp3`
* **Chapter-Based Output**: Organizes processed audio files into distinct chapter folders.


## Audio Processing Folder Structure
Organize your files as follows
- **EnglishOnly/**: Stores all English MP3 audio files here.
- **Languages/**: Inside the Languages folder, create subdirectories for each language here; e.g., **YorubaPhrasebook/YorubaOnly** for Yoruba audio files.
- For the audios-videos matching sentences, create `Yoruba_english_french_phrasebook_sentences_list.txt` in **YorubaPhrasebook**, with sentences formatted as: `1) language1 | Language2 | Language3;` e.g., `1) Hello.| Ẹ ǹ lẹ́. | Salut.`
- Notice that the language separator is |
- If you only want to display two languages in the Video, then only write 1) language1 | Language2; Example: 6)  I am fine.| Mo wà dáadáa.

**Best Practice:** Duplicate the `assets` folder and rename it to `private_assets`. Place your private audio and text resources inside `private_assets` to keep them separate from public/shared assets. This helps you manage sensitive or proprietary files that should not be distributed with the public repository.

## Customizable Parameters

- **local_language_name = 'yoruba'**: Specify the target local language.
- **SILENCE_THRESH = 1.5**: Chuncks separated by silence longer than this are considered distinct. you can adjust this based on your project.
- **INNER_PAUSE_DURATION = 3**: Desired silence duration between audio chunks.
- **REPEAT_LOCAL_AUDIO = 2**: Set to `2` to repeat the local language audio twice, or `1` if already repeated.
- **FLAG_PAD = True**: Enable (`True`) or disable (`False`) extending silence between chunks.

### Speed / parallelism (Step 1)

If Step 1 audio generation feels slow, use the parallelized variant scripts:
- `step1_audio_processing_v2.py` (recommended)
- `step1_audio_mix.py` (compatibility wrapper around v2)

Tuning knobs:
- `AUDIO_MAX_WORKERS` (env) or `MAX_WORKERS` in `step0_config.py` controls how many audio files are processed concurrently.
- `START_ID` / `END_ID` (env) restricts the sentence IDs to process for quick test runs.
- Chapter MP3s are concatenated with ffmpeg concat (`-c copy`) when possible (very fast); it falls back to re-encode / pydub if ffmpeg fails.

Example:

```powershell
$env:AUDIO_MAX_WORKERS = 6
$env:START_ID = 1; $env:END_ID = 200
python step1_audio_processing_v2.py
```

# Step 2: Video Production Pipeline

This module creates videos by combining the prepared audio and text files with visual assets.


![Demo resulting video](assets/demo_images/demo_image2.png)

## Additional Folders For Video Production

- **`Ads_Images/`**: Stores images to be used as advertisements. An ad is inserted at a customizable interval. The interval frequency is adjustable.
- **`Backgrounds/`**: Contains all background images for the videos. The script cycles through these images sequentially, with one unique background assigned per chapter.

## Video Parameters:
- **`local_language_name`**: Defines the local language for the video content.
- **`font_name_short`**: Specifies the font used for text in the videos.
- **`logo_name`**: The psth of the logo file to be used in the videos.

## Video Production Features:

* **Text-Audio Synchronization**: Maps processed bilingual audio files to their corresponding sentences from the text file.

* **Automated Resizing**: Automatically resizes all background images to `VIDEO_RESOLUTION` (default: 1920x1080).

* **Ad Placement**: Allows a portion of the screen (the lower half) to be used for ads or other content.

* **Customizable Logo**: Positions the Resulam logo at the bottom left and right of the video.

* **Start Point Control**: Enables the user to specify a starting sentence number for video production, allowing for easy resumption of a paused task.

## HD output vs. safe text area

- The final video output size is controlled by `VIDEO_RESOLUTION` (default `1920x1080`, i.e. HD).
- The captions use a smaller *safe area* inside that frame so they don't collide with the header row (language + `Please Support Resulam`) and the bottom logos:
  - Caption width budget: `VIDEO_RESOLUTION[0] - 200` (1920 - 200 = `1720px`)
  - Caption height budget: `VIDEO_RESOLUTION[1] - y_start - bottom_safe` (1080 - 120 - 140 = `820px`)
- Those numbers are *layout limits*, not the exported resolution.

## Long captions: wrapping + shrink-to-fit

`step2_video_production.py` renders captions with the bundled `assets/Fonts/CharisSIL-*.ttf` via Pillow. It then uses two mechanisms to keep text on-screen:

1. **Wrap to width** (`make_text_clip(..., size=(max_width, None))`)
   - Measures text in pixels and wraps by words.
   - If a single `word` is too long (no spaces), it is split into smaller chunks so it still fits.
   - Explicit newlines in the input are preserved.

2. **Shrink-to-fit height** (`fit_caption_block(...)`)
   - Used for the 3-line caption stack (Local + English + French).
   - Renders the whole stack at the requested font size, measures the total stacked height (+ gaps), and if it exceeds the height budget it reduces the font size by ~10% and tries again.
   - **Important behavior:** the stack uses *one font size for all three lines*, so if only the middle line is extremely long, *all three lines* will shrink until the **total** height fits.
   - Stops shrinking at `MIN_CAPTION_FONT_SIZE` from `step0_config.py` (optional; default is `18`). If the text still can't fit at the minimum, the stack may still overflow and you should lower `MIN_CAPTION_FONT_SIZE`, shorten the text, or increase the available layout budget.

### Performance / overhead

Shrink-to-fit is only extra work when shrinking is actually needed:
- If the stack fits at the requested font size, it renders once (no extra overhead).
- If it doesn't fit, it re-renders the 3 caption lines multiple times while searching for a size that fits. In worst-case `extremely long` triplets this can add noticeable time (multiple re-renders per sentence).
- The cost is proportional to the number of tries: each try rasterizes 3 text clips (Local/English/French), so 10 tries means ~30 text renders for that sentence.

### Demo: long triplet rendering

Use `tools/test_long_triplet_render.py` to see the behavior on long triplets:

```powershell
python .\tools\test_long_triplet_render.py
```

It writes `test_long_triplet_*_shrink.mp4` (and one `_noshrink.mp4` comparison) into the current language/mode output folder (same `VIDEO_OUT_DIR` used by `step2_video_production.py`).



![demo image 2](assets/demo_images/demo_image3.png)

[Amazon link to the book series](https://www.amazon.com/dp/B099TNLQL2)

# Step 3: Video Chunks Processor

This script is a post-production tool that combines multiple individual videos into larger, chapter-based video files.

# Step 4: Normalize combined videos (Udemy)

`step4_udemy_normalization.py` takes the combined chapter videos from Step 3 and creates normalized outputs under:

- `.../Results_Videos/<Mode>/<Language>_Chapters_Combined/normalized_output/`

This is the input folder for Step 5.

# Step 5: Add background music

`step5_add_background_music_new.py` overlays a background music track under each normalized chapter file:

- Music file: `.../Languages/<Language>Phrasebook/<MUSIC_FILENAME>` (default: `<language>_music_background.mp3`)
- Output folder: `.../Results_Videos/<Mode>/<Language>_Chapters_Combined/normalized_with_bg/`
- Volume: `MUSIC_GAIN_DB` in `step0_config.py` (more negative = quieter music)

How it works (high-level):
1. Loads the normalized file audio and the music file with `pydub.AudioSegment`.
2. Lowers the music volume using `MUSIC_GAIN_DB` from `step0_config.py` (default `-25.0`), loops it to match the speech duration, and overlays it under the speech.
3. For video files: exports a temporary WAV then uses `ffmpeg` to mux the original video stream with the new mixed audio (`-c:v copy`, `-shortest`).

## Quick benchmark and tuning

- To run a quick ffmpeg micro-benchmark on one sample mp3:

```powershell
python tools\benchmark_pipeline.py --step normalize --sample 1
```

- To tune parallelism before a full run, set env vars in PowerShell like:

```powershell
$env:MAX_WORKERS = 4; $env:FFMPEG_THREADS = 2
python step0_main_pipeline.py
```

Adjust `MAX_WORKERS` and `FFMPEG_THREADS` so that `MAX_WORKERS * FFMPEG_THREADS <=` number of CPU cores on your machine.
