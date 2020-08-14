from pathlib import Path

import gtts
import pyttsx3
import vlc


def tts(text, offline=False, **kwargs):
    if offline:
        tts_offline(text)
    else:
        try:
            tts_online(text, **kwargs)
        except gtts.gTTSError:
            tts_offline(text)


def tts_online(text, **kwargs):
    if kwargs.get("names"):
        file_path = Path("detection_code", "audio_files", f"""{"-".join(kwargs.get('names'))}.mp3""")
        kwargs.pop("names")
    _tts = gtts.gTTS(text, **kwargs)
    if not file_path.exists():
        _tts.save(str(file_path))
        print("saving file", str(file_path))
    p = vlc.MediaPlayer(str(file_path))
    p.play()


def tts_offline(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
