from pathlib import Path
import sys, os
# ensure project root is on sys.path so local modules can be imported when run from tools/
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
import step2_video_production as s2
from moviepy.editor import ColorClip, CompositeVideoClip

out_dir = s2.VIDEO_OUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)

# Random long sample sentence for testing wrapping / sizing
long_sentence = (
    "This is a deliberately long test sentence designed to exercise the text wrapping "
    "and rendering logic. It contains punctuation, several clauses, and enough length "
    "to require wrapping across multiple lines so we can validate layout, font metrics, "
    "and spacing behavior when using the Pillow-based renderer with Charis SIL. Keep adding more sentences so I see if at the end of the day, everything will fitl in one single view"
)

# Create background and rendered text clip using the module's utility
duration = 4
bg = ColorClip(s2.VIDEO_RESOLUTION, color=(10, 10, 30)).set_duration(duration)

txt_clip = s2.make_text_clip(long_sentence, font=s2.FONT_PATH,
                             fontsize=int(s2.DEFAULT_FONT_SIZE * 0.8), color="white",
                             size=(s2.VIDEO_RESOLUTION[0] - 200, None))
txt_clip = txt_clip.set_position(("center", "center")).set_duration(duration)

comp = CompositeVideoClip([bg, txt_clip]).set_duration(duration)

out_path = out_dir / "test_long_sentence_1.mp4"
print("Writing test video to:", out_path)
comp.write_videofile(str(out_path), fps=s2.FRAME_RATE, codec="libx264", audio=False, threads=1)
print("Done")
