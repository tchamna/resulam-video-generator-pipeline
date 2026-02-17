import subprocess
import sys
from pathlib import Path
import os


def _configure_stdio_utf8() -> None:
    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream and hasattr(stream, "reconfigure"):
            try: stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception: pass

_configure_stdio_utf8()

_ROOT = Path(__file__).resolve().parent
_EXCEL = "African_Languages_Phrasebook_GuideConversationExpressionsUsuelles_TabularPrototype_bon.xlsx"


def _find_excel() -> Path | None:
    for p in [_ROOT / "private_assets" / _EXCEL,
              *Path("G:/My Drive").glob(f"Mb*/{_EXCEL}"),
              *Path("G:/My Drive").glob(f"Mb*/Livres Nufi/{_EXCEL}")]:
        try:
            if p.exists(): return p
        except Exception: pass
    return None


def _regen_sentence_lists(excel: Path) -> None:
    import step0_config as cfg
    lang = getattr(cfg, "LANGUAGE", "").title()
    out = _ROOT / ("private_assets" if os.getenv("USE_PRIVATE_ASSETS", "1") == "1" else "assets") / "Languages"
    cmd = [sys.executable, str(_ROOT / "tools/generate_sentence_lists.py"),
           "--excel", str(excel), "--out-dir", str(out)]
    if lang:
        cmd += ["--sheets", lang]
    print(f"\n🔄 generate_sentence_lists.py (sheet: {lang or 'ALL'})\n   Excel : {excel}\n   Output: {out}\n")
    subprocess.run(cmd, check=True)
    print("✅ Sentence lists updated.\n")


def check_excel_freshness() -> None:
    print("\n" + "="*60)
    print("📋 PHRASEBOOK DATA CHECK")
    print("="*60)
    excel = _find_excel()
    print(f"  Excel: {excel or 'NOT FOUND'}")
    print(f"  Flow : docx → guidedeconversationphrasebook_processing.py → Excel")
    print(f"         Excel → generate_sentence_lists.py → TXT → pipeline")
    print("-"*60)

    if input("\n❓ Is the Excel up to date? (y/n): ").strip().lower() not in ("y", "yes"):
        print(f"\n⚠️  Run first:\n  {sys.executable} tools/guidedeconversationphrasebook_processing.py")
        sys.exit(0)

    if input("❓ Regenerate sentence list TXTs? (y/n): ").strip().lower() in ("y", "yes"):
        if not excel:
            sys.exit("❌ Excel not found.")
        _regen_sentence_lists(excel)

# List of step scripts in order
steps = [
   
    "step1_audio_processing_v2.py",
    "audio_processing_normal_pace.py",  # Produces Results_Audios_normal_pace/normal_rythm
    "step2_video_production.py",
    "step3_combine_videos.py",
    "step4_udemy_normalization.py",
    "step5_add_background_music_new.py",
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
    # Check if user has the latest Excel file (unless skipped via config)
    import step0_config as cfg
    if not getattr(cfg, "SKIP_PIPELINE_CHECKS", False):
        check_excel_freshness()
    
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
