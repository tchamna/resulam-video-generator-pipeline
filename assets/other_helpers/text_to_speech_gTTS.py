from gtts import gTTS

text = """
This Duala Audio Phrasebook was created by Doctor Tchamna and Iyo Ngobo,
and narrated by Muledi Dubé Dubé.

Get the physical book on Amazon,
and follow along as you listen.

This phrasebook is part of the African Languages Series,
first written in Nufi,
and now translated into many African languages.

Support the preservation of our mother tongues
by helping Resulam.
"""

tts = gTTS(text=text, lang="en")
tts.save("duala_phrasebook_intro.mp3")
