# from pydub import AudioSegment
# import os
# from natsort import natsorted

# # Define your folder path
# Chap_num = "Chap21"
# folder_path = fr"G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\Languages\ChichewaPhrasebook\bilingual_sentences\{Chap_num}"

# # Get all MP3 files and sort them naturally
# mp3_files = [f for f in os.listdir(folder_path) if f.endswith('.mp3')]
# mp3_files = natsorted(mp3_files)

# # Combine all audio files
# combined = AudioSegment.empty()
# for file in mp3_files:
#     audio = AudioSegment.from_mp3(os.path.join(folder_path, file))
#     combined += audio

# # Export the final combined audio
# output_path = os.path.join(folder_path, f"{Chap_num}_combined.mp3")
# combined.export(output_path, format="mp3")

# print(f"✅ Combined audio saved to: {output_path}")
###############################################
import os
from moviepy.editor import *
from natsort import natsorted

voice_folder = r"D:\moviepy_temp\Duala"
background_path = r"D:\Camtasia\Swahili 500 expressions.tscproj\Aakash Gandhi - Kiss the Sky.mp3"
output_folder = os.path.join(voice_folder, "mixed_output")
os.makedirs(output_folder, exist_ok=True)

# Load and prepare background music
try:
    bg_music = AudioFileClip(background_path)
except Exception as e:
    print(f"Error loading background music: {e}")
    exit()

bg_music = bg_music.volumex(0.1)  # Lower volume significantly (e.g., to 10%)

# Process each voice or video file in natural sort order
for filename in natsorted(os.listdir(voice_folder)):
    if filename.endswith(('.mp3', '.mp4', '.avi', '.mov')):
        if "Kiss the Sky" in filename:
            continue

        file_path = os.path.join(voice_folder, filename)
        
        # Determine if the file is audio or video and load accordingly
        if filename.endswith(('.mp4', '.avi', '.mov')):
            clip = VideoFileClip(file_path)
            voice_audio = clip.audio
        else: # .mp3
            clip = None
            voice_audio = AudioFileClip(file_path)

        # Loop and trim background music to match voice length
        loops = int(voice_audio.duration // bg_music.duration) + 1
        bg_looped = concatenate_audioclips([bg_music] * loops)
        bg_looped = bg_looped.set_duration(voice_audio.duration)

        # Apply fade in/out to background music
        fade_duration = min(3, voice_audio.duration * 0.1)  # 3 seconds or 10% of length
        bg_looped = bg_looped.audio_fadein(fade_duration).audio_fadeout(fade_duration)

        # Mix the voice audio with the looped background music
        mixed_audio = CompositeAudioClip([voice_audio, bg_looped])
        
        # Set the audio for the clip and export
        if clip:
            final_clip = clip.set_audio(mixed_audio)
            output_path = os.path.join(output_folder, f"mixed_{filename}")
            # Add remove_temp=False to prevent the PermissionError
            final_clip.write_videofile(output_path, codec="libx264", remove_temp=False)
            print(f"✅ Exported video: {output_path}")
        else:
            output_path = os.path.join(output_folder, f"mixed_{filename}")
            mixed_audio.write_audiofile(output_path, codec="mp3")
            print(f"✅ Exported audio: {output_path}")