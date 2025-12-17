from pathlib import Path
import sys
# ensure project root is importable
root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

import step2_video_production as s2
from moviepy.editor import ColorClip, CompositeVideoClip

out_dir = s2.VIDEO_OUT_DIR
out_dir.mkdir(parents=True, exist_ok=True)

def repeat_text(base: str, n: int) -> str:
    return (" ".join([base] * n)).strip()

# Create several long triplet sentences (English | Ewondo | French)
triplets = [
    (
        "This is an intentionally long English sentence designed to test wrapping behavior and to include commas, semicolons; and other punctuation marks so we can verify line breaking.",
        "Masóma, múnáye múnáte! Eyi ke a lɛ́ɛ̀, na bèndèlé, na marché é, mɛnɡá mɛnɡá; ɛ́ nɔ́ɲɔ́.",
        "Ceci est une phrase française longue destinée à tester le comportement du retour à la ligne et la mise en forme du texte pour s'assurer que tout s'affiche correctement."
    ),
    (
        "Another very long English example that includes parenthetical remarks (for instance, to test inline wrapping) and a mix of short and veryverylongwordsthatmightchallengewrapping algorithms.",
        "Ewondo exemple très long qui inclut des remarques entre parenthèses (par exemple, pour tester le retour à la ligne) et une combinaison de mots courts et de motsextrêmementlongsquelepourraientposerdesproblèmes.",
        "Une deuxième phrase française particulièrement longue, remplie de mots descriptifs, d'adjectifs et de clauses subordonnées pour pousser le rendu au-delà d'une seule ligne."
    ),
    (
        # Deliberately extreme triplet to demonstrate automatic shrinking to fit the frame.
        repeat_text(
            "This is a very long English caption designed to overflow at the default font size so we can demonstrate shrink-to-fit.",
            12,
        ),
        repeat_text(
            "This is a very long local-language caption designed to overflow at the default font size so we can demonstrate shrink-to-fit.",
            12,
        ),
        repeat_text(
            "Ceci est une longue phrase francaise concue pour depasser la hauteur disponible afin de demontrer la reduction automatique de la police.",
            12,
        ),
    ),
]

def main() -> None:
    # Render each triplet as a stacked text block (with shrink-to-fit)
    for i, (eng, local, fr) in enumerate(triplets, start=1):
        duration = 6
        bg = ColorClip(s2.VIDEO_RESOLUTION, color=(20, 30, 40)).set_duration(duration)

        fs_req = int(s2.DEFAULT_FONT_SIZE * 0.85)
        max_width = s2.VIDEO_RESOLUTION[0] - 200
        y_start = 120
        bottom_safe = 140
        max_height = s2.VIDEO_RESOLUTION[1] - y_start - bottom_safe
        gap = 10
        min_fs = int(getattr(s2.cfg, "MIN_CAPTION_FONT_SIZE", 18))

        # Shrink-to-fit render (uses the same helper as the main pipeline).
        fs_used, (local_clip, eng_clip, fr_clip) = s2.fit_caption_block(
            [(local, "white"), (eng, "yellow"), (fr, "white")],
            font=s2.FONT_PATH,
            fontsize=fs_req,
            max_width=max_width,
            max_height=max_height,
            gap=gap,
            min_fontsize=min_fs,
        )

        total_h = local_clip.h + eng_clip.h + fr_clip.h + gap * 2
        print(f"[triplet {i}] requested_fs={fs_req} -> used_fs={fs_used} | total_h={total_h} | max_h={max_height}")

        y = y_start
        local_clip = local_clip.set_position(("center", y)).set_duration(duration)
        y += local_clip.h + gap
        eng_clip = eng_clip.set_position(("center", y)).set_duration(duration)
        y += eng_clip.h + gap
        fr_clip = fr_clip.set_position(("center", y)).set_duration(duration)

        comp = CompositeVideoClip([bg, local_clip, eng_clip, fr_clip]).set_duration(duration)
        out_path = out_dir / f"test_long_triplet_{i}_shrink.mp4"
        print("Writing", out_path)
        comp.write_videofile(str(out_path), fps=s2.FRAME_RATE, codec="libx264", audio=False, threads=1)

        # For the last (extreme) triplet, also render a "no shrink" version for comparison.
        if i == len(triplets):
            local_ns = s2.make_text_clip(local, font=s2.FONT_PATH, fontsize=fs_req, color="white",
                                         size=(max_width, None))
            eng_ns = s2.make_text_clip(eng, font=s2.FONT_PATH, fontsize=fs_req, color="yellow",
                                       size=(max_width, None))
            fr_ns = s2.make_text_clip(fr, font=s2.FONT_PATH, fontsize=fs_req, color="white",
                                      size=(max_width, None))

            y = y_start
            local_ns = local_ns.set_position(("center", y)).set_duration(duration)
            y += local_ns.h + gap
            eng_ns = eng_ns.set_position(("center", y)).set_duration(duration)
            y += eng_ns.h + gap
            fr_ns = fr_ns.set_position(("center", y)).set_duration(duration)

            comp_ns = CompositeVideoClip([bg, local_ns, eng_ns, fr_ns]).set_duration(duration)
            out_path_ns = out_dir / f"test_long_triplet_{i}_noshrink.mp4"
            print("Writing (no shrink)", out_path_ns)
            comp_ns.write_videofile(str(out_path_ns), fps=s2.FRAME_RATE, codec="libx264", audio=False, threads=1)

    print("All done")


if __name__ == "__main__":
    main()
