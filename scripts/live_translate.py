#!/usr/bin/env python3
"""Live Dutch -> English captions from this laptop's microphone: Dutch left, English right.

The mic is cut into utterances at pauses, or at the quietest moment of 12 s. Nothing heavy runs on
this machine's CPU: it is always busy, and Whisper on it was too slow or too poor (2026-11-09).
Each clip is written down by Whisper on this machine's NVIDIA GPU when it has one (the home
laptop, run with the whisperX venv), else by the home Tiiny's Qwen3-ASR through the HAI OS owner
door on the tailnet; the Tiiny's chat model translates either way. The audio is saved beside the
captions log.

    python3 scripts/live_translate.py               # Ctrl+C stops; stdlib only, the Tiiny hears
    .venv/bin/python scripts/live_translate.py      # the home laptop: its GPU hears
    python3 scripts/live_translate.py --door https://<box>.<tailnet>.ts.net:18710
    python3 scripts/live_translate.py --selftest
"""

import argparse
import array
import io
import json
import math
import queue
import re
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import wave
from pathlib import Path
import urllib.error
import urllib.request

DOOR_FILE = Path.home() / ".config/hai/door"  # one line: the owner door's https URL
DOOR = ""  # set from --door or DOOR_FILE when the script starts
# One request at a time: a transcription and a translation in flight together took 8 to 40 s
# each on the Tiiny, one after the other 3.4 s and 1 to 2 s (2026-11-09)
BOX = threading.Lock()
RATE = 16000  # Qwen3-ASR hears a 48 kHz wav wrong; 16 kHz mono s16 is what it expects
FRAME = RATE // 10  # 100 ms of samples
LOG = Path.home() / ".local/state/live_translate" / time.strftime("%Y-%m-%d.log")
# past Latin-1 and punctuation: Dutch, English and French need no more, Polish or Czech do
NON_LATIN = re.compile(r"[^\x00-\u00ff\u2000-\u206f]")
DUTCH = set("de het een en is dat van ik je niet wat op zijn er maar ook dan we wij hij zij"
            " met voor als naar nog wel om bij dus kan heb hebben zo moet waar".split())
ENGLISH = set("the and is that of to it you not what on are but also then we he she with for"
              " as at this have i'm i my our your".split())
GERMAN = set("der die das und ist nicht ich mit sie auf ein eine auch sind wir für dass"
             " nach noch aber wie oder bei zu den dem des wenn kann haben so muss wo".split())
# Spanish, Italian, French; "de" left out, it is Dutch too
ROMANCE = set("para que la el los las por con una literalmente abbiamo detto il che non per"
              " le les est une pas avec pour".split())
# Danish, Norwegian, Swedish: the Tiiny drifted into them mid-clip (2026-11-09); "en" left out
NORDIC = set("og at det er som ikke til jeg har att och är inte jag på av för".split())
TRANSLATE_PROMPT = ("Translate the Dutch (Flemish) meeting transcript to English. It comes from speech"
                    " recognition and may contain misheard words: translate the intended meaning."
                    " Reply with the English only.")


def rms(frame: bytes) -> float:
    samples = array.array("h", frame)
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


class Chunker:
    """Groups 100 ms frames into utterances: ends one after `pause` quiet frames, or at `longest`.
    At `longest` the clip is cut at its quietest frame in the second half and the rest starts the
    next one: in a busy room speech rarely pauses long enough, and a cut mid-word garbled the
    words on both sides (2026-11-09)."""

    def __init__(self, threshold: float, pause: int = 7, shortest: int = 5, longest: int = 80):
        self.threshold, self.pause, self.shortest, self.longest = threshold, pause, shortest, longest
        self.frames: list[bytes] = []
        self.levels: list[float] = []

    def feed(self, frame: bytes) -> bytes | None:
        level = rms(frame)
        if not self.frames and level <= self.threshold:
            return None
        self.frames.append(frame)
        self.levels.append(level)
        quiet = next((i for i, x in enumerate(reversed(self.levels)) if x > self.threshold),
                     len(self.levels))
        if quiet >= self.pause:
            cut = len(self.frames)
        elif len(self.frames) >= self.longest:
            half = len(self.levels) // 2
            cut = half + min(range(len(self.levels) - half), key=lambda i: self.levels[half + i]) + 1
        else:
            return None
        clip, levels = b"".join(self.frames[:cut]), self.levels[:cut]
        self.frames, self.levels = self.frames[cut:], self.levels[cut:]
        return clip if sum(x > self.threshold for x in levels) >= self.shortest else None


def as_wav(pcm: bytes) -> bytes:
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return out.getvalue()


def post(path: str, body: bytes, content_type: str, timeout: float = 60) -> dict:
    req = urllib.request.Request(DOOR + path, data=body, headers={"content-type": content_type})
    try:
        with BOX, urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return {"error": json.load(e).get("error", {})}
        except ValueError:
            return {"error": {"message": f"HTTP {e.code}"}}


def transcribe(pcm: bytes, language: str = "") -> str:
    boundary = "liveTranslateBoundary"
    field = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n"
             f"{language}\r\n") if language else ""
    body = (f"{field}--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"u.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + as_wav(pcm) + f"\r\n--{boundary}--\r\n".encode()
    answer = post("/v1/audio/transcriptions", body, f"multipart/form-data; boundary={boundary}")
    if "error" in answer:
        raise RuntimeError(answer["error"].get("message", "transcription failed"))
    return answer.get("text", "").strip()


def columns(dutch: str, english: str, width: int) -> list[str]:
    """One utterance as aligned rows: Dutch wrapped in the left half, English in the right."""
    half = max((width - 3) // 2, 10)
    left, right = textwrap.wrap(dutch, half) or [""], textwrap.wrap(english, half) or [""]
    left += [""] * (len(right) - len(left))
    right += [""] * (len(left) - len(right))
    return [f"\033[2m{a:<{half}}\033[0m │ \033[1;32m{b}\033[0m" for a, b in zip(left, right)]


def heard_as(text: str) -> str:
    """"nl", "en" or "" (anything else): Latin script, then whose function words dominate."""
    if not text or NON_LATIN.search(text):
        return ""
    words = re.findall(r"[a-zà-ÿ']+", text.lower())
    nl, en, de, rom, nor = (sum(w in vocab for w in words)
                            for vocab in (DUTCH, ENGLISH, GERMAN, ROMANCE, NORDIC))
    if en > nl and en > de:
        return "en"
    return "nl" if nl > max(de, rom, nor) or (nl == de == rom == nor == 0 and len(words) < 4) else ""


def keep_heard(text: str) -> str:
    """The sentences heard as Dutch or English: the Tiiny switches to German, Danish or Polish
    for a sentence in the middle of a Dutch clip (2026-11-09)."""
    return " ".join(s for s in re.split(r"(?<=[.!?])\s+", text) if heard_as(s)).strip()


def translate(dutch: str, model: str, before: str = "") -> str:
    """The Tiiny's chat model, thinking off: 1 to 2 s a line, and it reads past misheard words
    where opus-mt translated them literally (2026-11-09). `before`, the Dutch just said, is only
    context: a clip often starts mid-sentence. It sits in the system message: put in the user
    message with a heading, the model once translated it along with the heading (2026-11-09)."""
    system = TRANSLATE_PROMPT + (f" For context only, never to translate, the Dutch said just before"
                                 f" was: «{before}»" if before else "")
    body = json.dumps({"model": model, "max_tokens": 400, "temperature": 0,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": dutch}]}).encode()
    answer = post("/v1/chat/completions", body, "application/json", timeout=120)
    if "error" in answer:
        raise RuntimeError(answer["error"].get("message", "translation failed"))
    return answer["choices"][0]["message"]["content"].strip()


def asr_choice(asked: str, gpus: int) -> str:
    """Which ASR writes the clips down: the GPU's Whisper when there is one, never this CPU."""
    if asked == "gpu" and not gpus:
        raise SystemExit("--asr gpu: no CUDA GPU here (run it with the whisperX venv's python)")
    return asked if asked != "auto" else ("gpu" if gpus else "tiiny")


def cuda_gpus() -> int:
    try:
        import ctranslate2  # the whisperX venv's; absent under the system python3
    except ImportError:
        return 0
    return ctranslate2.get_cuda_device_count()


def gpu_transcriber(language: str, size: str = "large-v3-turbo"):
    """faster-whisper on the GPU only. int8_float16 so the model fits beside the local
    model's ~3.4 GB on the home laptop's 6 GB card (not yet measured there). large-v3-turbo by
    default; large-v3, its full-size decoder, may hear Dutch better at about half the speed."""
    import numpy
    from faster_whisper import WhisperModel

    model = WhisperModel(size, device="cuda", compute_type="int8_float16")
    hot = Path.home() / ".local/share/whisperx/hotwords.txt"  # meeting.sh's names and terms
    words = " ".join(hot.read_text().split()) if hot.is_file() else None

    def transcribe_gpu(pcm: bytes, _: str = "") -> str:
        audio = numpy.frombuffer(pcm, dtype=numpy.int16).astype(numpy.float32) / 32768
        segments, _info = model.transcribe(audio, language=language or None, beam_size=5,
                                           condition_on_previous_text=False, hotwords=words)
        return " ".join(s.text.strip() for s in segments).strip()
    return transcribe_gpu


def hear(clips: queue.Queue, heard: queue.Queue, language: str, write=None) -> None:
    """Writes each clip down: the GPU's Whisper (`write`), or the Tiiny, 3 to 7 s per 12 s
    clip. The Tiiny ignores `language` and now and then hears Flemish as Chinese, Spanish or
    German: such a clip is dropped, never sent to this machine's CPU."""
    while True:
        pcm = clips.get()
        try:
            text = keep_heard((write or transcribe)(pcm, language))
        except (RuntimeError, OSError) as e:
            print(f"\033[31m[transcription: {e}]\033[0m", flush=True)
            continue
        if lang := heard_as(text):
            heard.put((text, lang))


def caption(heard: queue.Queue, model: str) -> None:
    """Translates on the Tiiny and prints. Lines that piled up while a translation ran go in one
    request, so a slow answer delays the captions once instead of adding up."""
    recent: list[str] = []
    while True:
        lines = [heard.get()]
        while not heard.empty():
            lines.append(heard.get())
        dutch = " ".join(text for text, _ in lines)
        if all(lang == "en" for _, lang in lines):  # someone spoke English
            english = dutch
        else:
            try:
                english = translate(dutch, model, " ".join(recent))
            except (RuntimeError, OSError, KeyError, IndexError) as e:
                english = f"[translation failed: {e}]"
        recent = (recent + [dutch])[-2:]
        for row in columns(dutch, english, shutil.get_terminal_size().columns):
            print(row, flush=True)
        print(flush=True)
        with LOG.open("a") as log:  # the session survives the window
            log.write(f"{time.strftime('%H:%M:%S')} NL {dutch}\n{time.strftime('%H:%M:%S')} EN {english}\n")


def main(language: str, model: str, asr: str, whisper: str) -> int:
    asr = asr_choice(asr, cuda_gpus())
    write = gpu_transcriber(language, whisper) if asr == "gpu" else None
    mic = subprocess.Popen(["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16",
                            "--raw", "-"], stdout=subprocess.PIPE)
    assert mic.stdout is not None
    stdout = mic.stdout
    LOG.parent.mkdir(parents=True, exist_ok=True)
    audio = LOG.parent / time.strftime("%Y-%m-%d %H%M.wav")
    record = wave.open(str(audio), "wb")
    record.setnchannels(1)
    record.setsampwidth(2)
    record.setframerate(RATE)

    def read() -> bytes:
        frame = stdout.read(FRAME * 2)
        record.writeframes(frame)  # rewrites the header each time: a killed window leaves a valid file
        return frame

    print(f"Hearing on the {'GPU (Whisper)' if write else 'Tiiny'}; saving the audio to {audio}", flush=True)
    # the quietest 100 ms of 2 s: people may already be talking when this starts
    floor = min(rms(read()) for _ in range(20))
    # 12 s, not 8: on room audio the box garbled more clips cut at 5 to 8 s (2026-11-09 A/B)
    chunker = Chunker(threshold=max(floor * 1.3, 150), longest=120)
    print(f"Listening (noise floor {floor:.0f}, speech above {chunker.threshold:.0f}). Ctrl+C stops.\n")
    clips: queue.Queue = queue.Queue()
    heard: queue.Queue = queue.Queue()
    threading.Thread(target=hear, args=(clips, heard, language, write), daemon=True).start()
    threading.Thread(target=caption, args=(heard, model), daemon=True).start()
    try:
        while frame := read():
            if clip := chunker.feed(frame):
                clips.put(clip)
    except KeyboardInterrupt:
        pass
    finally:
        mic.terminate()
        record.close()
        print(f"\nAudio saved: {audio}\nCaptions: {LOG}")
    return 0


def selftest() -> None:
    quiet, loud = array.array("h", [10] * FRAME).tobytes(), array.array("h", [3000] * FRAME).tobytes()
    c = Chunker(threshold=150)
    assert c.feed(quiet) is None and not c.frames, "leading silence is dropped"
    out = [c.feed(f) for f in [loud] * 10 + [quiet] * 7]
    assert out[:-1] == [None] * 16 and len(out[-1] or b"") == 17 * FRAME * 2, "cut after a 0.7 s pause"
    assert [c.feed(f) for f in [loud] * 2 + [quiet] * 7][-1] is None, "a cough is not an utterance"
    assert sum(c.feed(loud) is not None for _ in range(80)) == 1, "a long monologue is cut at 8 s"
    dip = array.array("h", [400] * FRAME).tobytes()  # a breath: louder than silence, quietest here
    c = Chunker(threshold=150)
    out = [c.feed(f) for f in [loud] * 59 + [dip] + [loud] * 20]
    assert len(out[-1] or b"") == 60 * FRAME * 2, "cut at the quietest frame, not mid-word"
    assert len(c.frames) == 20, "the words after the cut start the next clip"
    rows = columns("een twee drie vier vijf zes", "one two", 24)
    assert len(rows) == 3 and all("│" in r for r in rows), "Dutch and English share rows"
    assert rows[0].index("│") == rows[2].index("│"), "the divider is aligned"
    assert heard_as("Dat kan net zo goed bouwen zijn, daar kan je tafels leveren.") == "nl"
    assert heard_as("Goedemiddag, allemaal.") == heard_as("Brussels Airport Company.") == "nl"
    assert heard_as("Hi, I'm the chief officer at the company and this is our plan.") == "en"
    assert heard_as("Das ist nicht so, und wir haben auch keine Zeit dafür.") == "", "German"
    assert heard_as("las autoridades afirman que esto ha sido posible") == "", "Spanish"
    assert heard_as("literalmente para") == heard_as("abbiamo detto.") == "", "short Spanish, Italian"
    assert heard_as("Dank u wel.") == "nl"
    assert heard_as("به دیزد.و در مورد این تکنیک") == "", "Persian"
    assert asr_choice("auto", 1) == "gpu" and asr_choice("auto", 0) == "tiiny"
    assert asr_choice("tiiny", 1) == "tiiny"
    try:
        asr_choice("gpu", 0)
        raise AssertionError("no GPU: --asr gpu refuses, never this CPU")
    except SystemExit:
        pass
    assert heard_as("Vi såg verkligen att det är en fantastisk och sort dåsätt.") == "", "Swedish"
    assert keep_heard("Dat kan je tafels leveren. Moje powodowało to stąd, że on nie rozdera. Dank u wel.") \
        == "Dat kan je tafels leveren. Dank u wel.", "a foreign sentence inside a Dutch clip goes"
    assert heard_as("哎呀，我天哪！") == heard_as("مرحبا بكم") == "", "Chinese, Arabic"
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--language", default="nl", help="spoken language (Whisper holds to it; the Tiiny ignores it today)")
    p.add_argument("--asr", choices=("auto", "gpu", "tiiny"), default="auto",
                   help="who writes the Dutch down: this machine's CUDA GPU, else the Tiiny (auto)")
    p.add_argument("--whisper-model", default="large-v3-turbo",
                   help="the GPU's Whisper: large-v3-turbo (default) or large-v3, slower, maybe better Dutch")
    p.add_argument("--door", help=f"HAI OS owner door URL (default: the line in {DOOR_FILE})")
    p.add_argument("--model", default="box/tiiny/Qwen/Qwen3.6-35B-A3B", help="chat model translating on the box")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        try:
            DOOR = (args.door or DOOR_FILE.read_text()).strip().rstrip("/")
        except FileNotFoundError:
            sys.exit(f"No door: pass --door or write its URL into {DOOR_FILE}")
        sys.exit(main(args.language, args.model, args.asr, args.whisper_model))
