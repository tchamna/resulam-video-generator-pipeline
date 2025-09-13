

import librosa
import soundfile as sf
import noisereduce as nr
import os

import subprocess
from pathlib import Path

def ffmpeg_denoise(input_file, output_file):
    cmd = [
        "ffmpeg", "-y", "-i", str(input_file),
        "-af", "afftdn=nr=20:nt=w:om=o",  # spectral denoise
        str(output_file)
    ]
    subprocess.run(cmd, check=True)
    print(f"✅ Denoised (ffmpeg): {output_file}")


def reduce_noise_from_audio(input_file, output_file):
    """
    Reduces noise from an audio file and saves the result.
    
    Args:
        input_file (str): The path to the input audio file.
        output_file (str): The path where the denoised audio will be saved.
    """
    try:
        # Load the audio file
        # sr=None keeps the original sampling rate of the audio
        y, sr = librosa.load(input_file, sr=None)

        # Apply noise reduction
        # The 'y' is the audio signal, and 'sr' is the sampling rate
        # stationary=True assumes the noise is consistent throughout the audio
        reduced_noise_audio = nr.reduce_noise(y=y, 
                                              sr=sr, 
                                              stationary=True,n_fft=2048,)

        # Save the denoised audio to a new file
        # 'sf.write' handles various audio formats (e.g., .wav, .flac)
        sf.write(output_file, reduced_noise_audio, sr)
        print(f"Noise reduction complete! Denoised audio saved to {output_file}")
    
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
# Replace 'your_audio.mp3' with the path to your input audio file
# Replace 'denoised_audio.wav' with your desired output file name
input_file = r'G:\My Drive\Data_Science\Resulam\Phrasebook_Audio_Video_Processing_production\Languages\KinyarwandaPhrasebook\NoisyAudio\kinyarwanda_4.mp3'
output_audio_path = "denoised_noise_reduce_"+ os.path.basename(input_file) 
output_audio_path_ffmpeg = "denoised_ffmpeg_"+ os.path.basename(input_file) 

# Call the function to perform the noise reduction
reduce_noise_from_audio(input_file, output_audio_path)
ffmpeg_denoise(input_file, output_audio_path_ffmpeg)
#############################################################################################
# import torchaudio
# from DeepFilterNet import DeepFilterNet

# # Load the audio model
# model = DeepFilterNet()

# # Load your audio file (the library may require specific formats, like .wav)
# # You may need to convert your .mp3 file to .wav first using a library like pydub
# audio_signal, sample_rate = torchaudio.load(input_file)

# # Process the audio with the AI model
# denoised_signal = model(audio_signal)

# # Save the output
# output_file = "denoised_duala_phrasebook.wav"
# torchaudio.save(output_file, denoised_signal, sample_rate)
# print(f"Denoised audio saved to {output_file}")