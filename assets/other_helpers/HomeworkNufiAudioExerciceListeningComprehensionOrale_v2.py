import os
import csv
from uuid import uuid4 # for unique temp filenames
import shutil

import random
from pathlib import Path
from moviepy.editor import (
    AudioFileClip,
    ImageClip,
    TextClip,
    CompositeVideoClip,
    CompositeAudioClip,
)

def parse_text_file_to_text_data(txt_path, audio_nufi_prefix, audio_en_prefix, audio_ext=".mp3"):
    text_data = []

    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            if "|" not in line:
                continue  # skip invalid lines

            try:
                parts = line.strip().split("|")
                id_part, english = parts[0].strip().split(")", 1)
                id_num = int(id_part.strip())
                nufi = parts[1].strip()
                french = parts[2].strip()

                entry = {
                    "id": id_num,
                    "english": english.strip(),
                    "nufi": nufi,
                    "french": french,
                    "audio_file": f"{audio_nufi_prefix}{id_num}{audio_ext}",
                    "english_audio_file": f"{audio_en_prefix}{id_num}{audio_ext}"
                }
                text_data.append(entry)
            except Exception as e:
                print(f"⚠️ Skipping line due to error: {e}\nLine: {line.strip()}")

    return text_data

def get_text_clip(text, start, duration, y_pixels, font=None, color="white", fontsize=80):
    return TextClip(
        text,
        fontsize=fontsize,
        color=color,
        font=font or DEFAULT_FONT,
        method='caption',
        size=(VIDEO_SIZE[0] - 200, None)  # narrower width to force wrapping
    ).set_position(("center", y_pixels)).set_start(start).set_duration(duration)


def create_video(entry, background_path, output_path, base_fontsize=100):
    try:
        nufi_audio_path = NUFI_AUDIO_DIR / entry["audio_file"]
        eng_audio_path = EN_AUDIO_DIR / entry["english_audio_file"]

        if not nufi_audio_path.exists():
            raise FileNotFoundError(f"Nufi audio file not found: {nufi_audio_path}")
        if not eng_audio_path.exists():
            raise FileNotFoundError(f"English audio file not found: {eng_audio_path}")

        nufi_audio = AudioFileClip(str(nufi_audio_path))
        eng_audio = AudioFileClip(str(eng_audio_path))

        nufi_dur = nufi_audio.duration
        eng_dur = eng_audio.duration

        nufi_1 = nufi_audio.set_start(0)
        nufi_2_start = nufi_dur + PAUSE
        nufi_2 = nufi_audio.set_start(nufi_2_start)
        nufi_3_start = nufi_2_start + nufi_dur + PAUSE
        nufi_3 = nufi_audio.set_start(nufi_3_start)
        eng_start = nufi_3_start + nufi_dur + PAUSE
        eng_clip = eng_audio.set_start(eng_start)

        total_duration = eng_start + eng_dur + FINAL_PAUSE
        audio = CompositeAudioClip([nufi_1, nufi_2, nufi_3, eng_clip])

        # Font size adjustment
        combined_text = f"{entry['nufi']} {entry['english']} {entry['french']}"
        total_chars = len(combined_text)
        if total_chars < 50:
            fontsize = base_fontsize
        elif total_chars < 80:
            fontsize = int(base_fontsize * 0.85)
        elif total_chars < 110:
            fontsize = int(base_fontsize * 0.75)
        else:
            fontsize = int(base_fontsize * 0.65)

        line_spacing = int(fontsize * 1.2)
        y_base = 150

        # Load background image with fallback
        try:
            print(f"🎨 Loading background image: {background_path.name}")
            if background_path.stat().st_size < 1024:
                raise ValueError("Background image too small or empty.")
            bg = ImageClip(str(background_path)).resize(height=VIDEO_SIZE[1], width=VIDEO_SIZE[0]).set_duration(total_duration)
        except Exception as e:
            print(f"⚠️ Failed to load background {background_path.name}: {e}")
            print(f"➡️ Using fallback background: {DEFAULT_BG_PATH.name}")
            bg = ImageClip(str(DEFAULT_BG_PATH)).resize(height=VIDEO_SIZE[1], width=VIDEO_SIZE[0]).set_duration(total_duration)

        clips = [bg]

        # Intro text
        clips.append(get_text_clip("Yū' Mfʉ́ɑ́'sí, Mfāhngə́ə́:", 0, nufi_2_start,
                                   y_pixels=100, color="white", fontsize=int(fontsize * 0.9)))
        clips.append(get_text_clip("Listen, repeat and translate", 0, nufi_2_start,
                                   y_pixels=100 + line_spacing, color="yellow", fontsize=fontsize))

        # Sentences
        clips.append(get_text_clip(entry["nufi"], nufi_2_start, total_duration - nufi_2_start,
                                   y_pixels=y_base, color="white", fontsize=fontsize))
        clips.append(get_text_clip(entry["english"], nufi_3_start, total_duration - nufi_3_start,
                                   y_pixels=y_base + line_spacing, color="yellow", fontsize=fontsize))
        clips.append(get_text_clip(entry["french"], nufi_3_start, total_duration - nufi_3_start,
                                   y_pixels=y_base + 2 * line_spacing, color="white", fontsize=fontsize))

        # Top & bottom text
        clips.append(get_text_clip("Please Support Resulam", 0, total_duration,
                                   y_pixels=20, color="yellow", fontsize=int(fontsize * 0.5)))
        clips.append(get_text_clip("www.resulam.com", 0, total_duration,
                                   y_pixels=VIDEO_SIZE[1] - 100, color="yellow", fontsize=int(fontsize * 0.5)))

        # Logos
        logo_left = ImageClip(logo_path).resize(height=80).set_position(("left", "bottom")).set_duration(total_duration)
        logo_right = ImageClip(logo_path).resize(height=80).set_position(("right", "bottom")).set_duration(total_duration)
        clips += [logo_left, logo_right]

        # Render video
        print(f"🎬 Rendering video: {output_path.name}")
        video = CompositeVideoClip(clips).set_audio(audio)
        video.write_videofile(
            str(output_path),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-audio.m4a",
            remove_temp=True
        )

    except Exception as e:
        print(f"❌ Error creating video for sentence {entry['id']}: {e}")
        if output_path.exists():
            output_path.unlink()  # remove partially written file


def resize_background_safe(img_path, target_size):
    clip = ImageClip(str(img_path)).resize(target_size)
    w, h = clip.size
    w_even = w if w % 2 == 0 else w + 1
    h_even = h if h % 2 == 0 else h + 1
    return clip.resize((w_even, h_even))

def resize_with_padding(image_path, target_size):
    img = ImageClip(str(image_path)).resize(height=target_size[1])
    padded = img.on_color(
        size=target_size,
        color=(0, 0, 0),  # or a background color that matches your aesthetic
        pos=('center', 'center')
    )
    return padded


def create_video(entry, background_path, output_path, base_fontsize=100):
    try:
        nufi_audio_path = NUFI_AUDIO_DIR / entry["audio_file"]
        eng_audio_path = EN_AUDIO_DIR / entry["english_audio_file"]

        if not nufi_audio_path.exists():
            raise FileNotFoundError(f"Nufi audio file not found: {nufi_audio_path}")
        if not eng_audio_path.exists():
            raise FileNotFoundError(f"English audio file not found: {eng_audio_path}")

        nufi_audio = AudioFileClip(str(nufi_audio_path))
        eng_audio = AudioFileClip(str(eng_audio_path))

        nufi_dur = nufi_audio.duration
        eng_dur = eng_audio.duration

        nufi_1 = nufi_audio.set_start(0)
        nufi_2_start = nufi_dur + PAUSE
        nufi_2 = nufi_audio.set_start(nufi_2_start)
        nufi_3_start = nufi_2_start + nufi_dur + PAUSE
        nufi_3 = nufi_audio.set_start(nufi_3_start)
        eng_start = nufi_3_start + nufi_dur + PAUSE
        eng_clip = eng_audio.set_start(eng_start)

        total_duration = eng_start + eng_dur + FINAL_PAUSE
        audio = CompositeAudioClip([nufi_1, nufi_2, nufi_3, eng_clip])

        combined_text = f"{entry['nufi']} {entry['english']} {entry['french']}"
        total_chars = len(combined_text)
        if total_chars < 50:
            fontsize = base_fontsize
        elif total_chars < 80:
            fontsize = int(base_fontsize * 0.85)
        elif total_chars < 110:
            fontsize = int(base_fontsize * 0.75)
        else:
            fontsize = int(base_fontsize * 0.65)

        line_spacing = int(fontsize * 1.2)
        y_base = 150
        
        
        try:
            print(f"🎨 Loading background image: {background_path.name}")
            if background_path.stat().st_size < 1024:
                raise ValueError("Background image too small or empty.")
            
            # bg = resize_background_safe(background_path, VIDEO_SIZE).set_duration(total_duration)
            bg = resize_with_padding(background_path, VIDEO_SIZE).set_duration(total_duration)

        except Exception as e:
            print(f"⚠️ Failed to load background {background_path.name}: {e}")
            print(f"➡️ Using fallback background: {DEFAULT_BG_PATH.name}")
            bg = ImageClip(str(DEFAULT_BG_PATH)).resize(
                height=VIDEO_SIZE[1],
                width=VIDEO_SIZE[0]
            ).set_duration(total_duration)


        clips = [bg]
        clips.append(get_text_clip("Yū' Mfʉ́ɑ́'sí, Mfāhngə́ə́:", 0, nufi_2_start,
                                   y_pixels=100, color="white", fontsize=int(fontsize * 0.9)))
        clips.append(get_text_clip("Listen, repeat and translate", 0, nufi_2_start,
                                   y_pixels=100 + line_spacing, color="yellow", fontsize=fontsize))
        clips.append(get_text_clip(entry["nufi"], nufi_2_start, total_duration - nufi_2_start,
                                   y_pixels=y_base, color="white", fontsize=fontsize))
        clips.append(get_text_clip(entry["english"], nufi_3_start, total_duration - nufi_3_start,
                                   y_pixels=y_base + line_spacing, color="yellow", fontsize=fontsize))
        clips.append(get_text_clip(entry["french"], nufi_3_start, total_duration - nufi_3_start,
                                   y_pixels=y_base + 2 * line_spacing, color="white", fontsize=fontsize))
        clips.append(get_text_clip("Please Support Resulam", 0, total_duration,
                                   y_pixels=20, color="yellow", fontsize=int(fontsize * 0.5)))
        clips.append(get_text_clip("www.resulam.com", 0, total_duration,
                                   y_pixels=VIDEO_SIZE[1] - 100, color="yellow", fontsize=int(fontsize * 0.5)))

        logo_left = ImageClip(logo_path).resize(height=80).set_position(("left", "bottom")).set_duration(total_duration)
        logo_right = ImageClip(logo_path).resize(height=80).set_position(("right", "bottom")).set_duration(total_duration)
        clips += [logo_left, logo_right]

        print(f"🎬 Rendering video: {output_path.name}")
        video = CompositeVideoClip(clips).set_audio(audio)

        # Generate safe filenames
        temp_audio_path = f"temp-audio-{uuid4().hex}.m4a"
        temp_output_path = output_path.with_suffix(".temp.mp4")

        video.write_videofile(
            str(temp_output_path),
            fps=FPS,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=temp_audio_path,
            remove_temp=True,
            preset="ultrafast",
            threads=4
        )

        # Rename only if success
        shutil.move(str(temp_output_path), str(output_path))
        print(f"✅ Created: {output_path.name}")

    except Exception as e:
        print(f"❌ Error creating video for sentence {entry['id']}: {e}")
        # Clean up partial files
        if output_path.exists():
            output_path.unlink()
        temp_output_path = output_path.with_suffix(".temp.mp4")
        if temp_output_path.exists():
            temp_output_path.unlink()

def generate_all(test_mode=True, start_id=None, end_id=None, backgrounds=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not backgrounds:
        raise RuntimeError("Backgrounds list must be provided.")

    current_chapter = None
    current_bg = None
    bg_index = -1
    log_rows = []

    filtered_data = [
        entry for entry in TEXT_DATA
        if (start_id is None or entry["id"] >= start_id) and
           (end_id is None or entry["id"] <= end_id)
    ]

    for i, entry in enumerate(filtered_data):
        english_text = entry["english"]

        if english_text.strip().lower().startswith("chapter"):
            if current_chapter != english_text:
                current_chapter = english_text
                bg_index = (bg_index + 1) % len(backgrounds)
                current_bg = backgrounds[bg_index]

        if test_mode:
            log_rows.append({
                "sentence_id": entry["id"],
                "chapter": current_chapter,
                "background_file": current_bg.name if current_bg else "None"
            })
        else:
            output = OUTPUT_DIR / f"{LANGUAGE.lower()}_sentence_{entry['id']}.mp4"
            try:
                create_video(entry, current_bg, output, base_fontsize=FONT_SIZE)
                print(f"✅ Created: sentence_{entry['id']}.mp4")
            except Exception as e:
                print(f"❌ Error with sentence {entry['id']}: {e}")

    if test_mode:
        csv_path = OUTPUT_DIR / "background_assignment_log.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["sentence_id", "chapter", "background_file"])
            writer.writeheader()
            writer.writerows(log_rows)
        print(f"📄 CSV log saved to: {csv_path}")

DEFAULT_BG_PATH = Path(r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\Backgrounds_Selected\chap2_darkblueBackground.jpg")

LANGUAGE = "Nufi"
FONT = r"C:\Users\tcham\OneDrive\Documents\Workspace_Codes\dictionnaire-nufi-franc-nufi\app\src\main\assets\fonts\CharisSIL-B.ttf"
DEFAULT_FONT = FONT

NUFI_AUDIO_DIR = Path(fr"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\Languages\{LANGUAGE}Phrasebook\{LANGUAGE}Only")
BACKGROUND_IMG_DIR = Path(r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\Backgrounds_Selected")
OUTPUT_DIR = Path(fr"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\Python_Scripts_Resulam_Phrasebooks_Audio_Processing\{LANGUAGE}TestResultat")
EN_AUDIO_DIR = Path(fr"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\EnglishOnly")

logo_path = r"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production_Backup\resulam_logo_resurrectionLangue.png"
txt_path = fr"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\Languages\{LANGUAGE}Phrasebook\{LANGUAGE.lower()}_english_french_phrasebook_sentences_list.txt"

FONT_SIZE = 100
TEXT_COLOR = "white"
VIDEO_SIZE = (1920, 1080)
PAUSE = 5
FINAL_PAUSE = 3
FPS = 24

NUFI_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
BACKGROUND_IMG_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EN_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

TEXT_DATA = parse_text_file_to_text_data(
    txt_path,
    audio_nufi_prefix=f"{LANGUAGE.lower()}_phrasebook_",
    audio_en_prefix="english_",
    audio_ext=".mp3"
)

if __name__ == "__main__":

    print("Starting video generation...")

    # If both start_id and end_id are None, the script will process all sentences in TEXT_DATA.
    generate_all(test_mode=False, start_id=1570, end_id=2030)
