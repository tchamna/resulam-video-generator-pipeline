import subprocess
import sys
from pathlib import Path
import os

# List of step scripts in order
steps = [
    # "step1_audio_processing.py",
    # "step2_video_production.py",
    # "step3_combine_videos.py",
    "step4_add_background_music.py",
]

def run_script(script):
    print(f"\n🚀 Running {script} ...\n{'='*50}")
    try:
        subprocess.run([sys.executable, script], check=True)
        print(f"✅ Completed {script}\n{'='*50}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error in {script} (exit code {e.returncode})")
        sys.exit(e.returncode)

if __name__ == "__main__":
    
    
    if "USE_PRIVATE_ASSETS" not in os.environ:
        os.environ["USE_PRIVATE_ASSETS"] = "1"

    USE_PRIVATE_ASSETS = os.getenv("USE_PRIVATE_ASSETS") == "1"

    
    base_dir = Path(__file__).parent

    for step in steps:
        script_path = base_dir / step
        if not script_path.exists():
            print(f"⚠️ Skipping {step}, file not found")
            continue
        run_script(str(script_path))

    print("\n🎉 All steps finished successfully!")
