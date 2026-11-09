#!/usr/bin/env python3
"""Prints the Whisper model meeting.sh should load for a recording: large-v3-turbo when it is
English, large-v3 otherwise (scripts/live_translate.py whisper_size). The language is the
`--language` among whisperX's arguments when one is given, else what Whisper turbo hears in the
recording's first speech, on the GPU only: meeting.sh calls this in its CUDA branch.

    python scripts/pick_whisper.py <audio> [whisperx args...]
    python scripts/pick_whisper.py --selftest
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from live_translate import whisper_size  # noqa: E402

FIRST_S = 180  # enough for three 30 s windows of speech after a quiet start


def asked_language(args: list[str]) -> str:
    for i, arg in enumerate(args):
        if arg == "--language" and i + 1 < len(args):
            return args[i + 1]
        if arg.startswith("--language="):
            return arg.split("=", 1)[1]
    return ""


def heard_language(audio: str) -> str:
    from faster_whisper import WhisperModel, decode_audio

    model = WhisperModel("large-v3-turbo", device="cuda", compute_type="int8_float16")
    # shortcut: decodes the whole file to keep its first minutes; cut with ffmpeg if hour-long files lag here
    pcm = decode_audio(audio)[:FIRST_S * 16000]
    language, probability, _ = model.detect_language(pcm, vad_filter=True, language_detection_segments=3)
    print(f"heard {language} ({probability:.2f})", file=sys.stderr)
    return language


def selftest() -> None:
    assert asked_language(["--diarize", "--language", "en"]) == "en"
    assert asked_language(["--language=fr"]) == "fr"
    assert asked_language(["--diarize"]) == "" and asked_language(["--language"]) == ""
    assert whisper_size("auto", asked_language(["--language", "en"])) == "large-v3-turbo"
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif len(sys.argv) >= 2:
        print(whisper_size("auto", asked_language(sys.argv[2:]) or heard_language(sys.argv[1])))
    else:
        sys.exit(__doc__)
