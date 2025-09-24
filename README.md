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

## Customizable Parameters

- **local_language_name = 'yoruba'**: Specify the target local language.
- **SILENCE_THRESH = 1.5**: Chuncks separated by silence longer than this are considered distinct. you can adjust this based on your project.
- **INNER_PAUSE_DURATION = 3**: Desired silence duration between audio chunks.
- **REPEAT_LOCAL_AUDIO = 2**: Set to `2` to repeat the local language audio twice, or `1` if already repeated.
- **FLAG_PAD = True**: Enable (`True`) or disable (`False`) extending silence between chunks.

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

* **Automated Resizing**: Automatically resizes all background images to a 1980x1080 pixel resolution.

* **Ad Placement**: Allows a portion of the screen (the lower half) to be used for ads or other content.

* **Customizable Logo**: Positions the Resulam logo at the bottom left and right of the video.

* **Start Point Control**: Enables the user to specify a starting sentence number for video production, allowing for easy resumption of a paused task.



![demo image 2](assets/demo_images/demo_image3.png)

[Amazon link to the book series](https://www.amazon.com/dp/B099TNLQL2)

# Step 3: Video Chunks Processor

This final script is a post-production tool that combines multiple individual videos into larger, chapter-based video files.
