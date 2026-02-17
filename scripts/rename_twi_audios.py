"""
Recursively rename audio files in a directory, replacing '-2' with '_' in filenames.
Example: 3-2.mp3 becomes 3_.mp3
"""

import os
from pathlib import Path


def rename_audio_files(directory: str) -> None:
    """
    Recursively traverse directory and rename audio files containing '-2' to '_'.
    
    Args:
        directory: Path to the directory to process
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"Error: Directory does not exist: {directory}")
        return
    
    if not dir_path.is_dir():
        print(f"Error: Not a directory: {directory}")
        return
    
    renamed_count = 0
    
    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(dir_path):
        for filename in files:
            # Check if filename contains '-2'
            if '-2' in filename:
                # Only process audio files
                if filename.lower().endswith(('.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a')):
                    old_path = Path(root) / filename
                    new_filename = filename.replace('-2', '_')
                    new_path = Path(root) / new_filename
                    
                    try:
                        old_path.rename(new_path)
                        print(f"✓ Renamed: {filename} -> {new_filename}")
                        renamed_count += 1
                    except Exception as e:
                        print(f"✗ Error renaming {filename}: {e}")
    
    print(f"\nTotal files renamed: {renamed_count}")


if __name__ == "__main__":
    # Path to the Twi audio directory
    source_dir = r"G:\My Drive\Mbú'ŋwɑ̀'nì\Livres Twi\Twi_AUDIO\TwiAudioPhrasebookCuts"
    rename_audio_files(source_dir)
