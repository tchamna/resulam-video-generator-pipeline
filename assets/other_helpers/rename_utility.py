

LOCAL_LANGUAGE = "bamoun"
import os   

def files_rename(directory, prefix="", suffix="", replace="", by="",
                 remove_first=0, remove_last=0,
                 lower_all=False, upper_all=True,
                 extensions=(".mp3", ".wav")):
    if isinstance(extensions, str):
        extensions = tuple(ext.strip() for ext in extensions.split(','))
    if lower_all and upper_all:
        print("Both lower_all and upper_all True. Defaulting to lowercase.")
        upper_all = False
    for filename in os.listdir(directory):
        # if not filename.startswith(LOCAL_LANGUAGE):
            original_filename = filename
            if lower_all:
                filename = filename.lower()
            elif upper_all:
                filename = filename.upper()
            try:
                new_name = filename.replace(replace, by)
                new_name = new_name[remove_first:]
                new_name = new_name[:len(new_name) - remove_last]
                new_name = prefix + new_name + suffix
            except Exception:
                new_name = filename

            if extensions is None or (isinstance(extensions, tuple) and original_filename.lower().endswith(extensions)):
                old_file = os.path.join(directory, original_filename)
                new_file = os.path.join(directory, new_name)
                try:
                    os.rename(old_file, new_file)
                    print(f"Renamed '{original_filename}' → '{new_name}'")
                except Exception as e:
                    print(f"❌ Rename failed '{original_filename}': {e}")
            else:
                print(f"Skipped '{original_filename}': extension mismatch")

directory = r"C:\Users\tcham\OneDrive\Documents\Workspace_Codes\AfricanLanguagesPhrasebook_Web\AfricanLanguagesPhrasebook\frontend\build\audio_files\audio_bamoun"

# nufi_phrasebook_3.mp3
# bamoun_phrasebook_3.mp3

files_rename(directory,
                    prefix="", suffix="",
                    replace="_padded", by="",
                    remove_first=0, remove_last=0,
                    lower_all=True, upper_all=False,
                    extensions=(".mp3", ".wav"))