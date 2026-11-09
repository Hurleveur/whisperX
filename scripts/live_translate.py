#!/usr/bin/env python3
"""Live Dutch -> English captions from this laptop's microphone: Dutch left, English right.

The mic is cut into utterances at pauses. The home Tiiny's Qwen3-ASR (through the HAI OS owner
door on the tailnet) writes each one down in Dutch; opus-mt-nl-en on this laptop translates that
text. The Tiiny cannot also translate: it serves one request at a time and transcribing already
keeps it busy, so a translation there waited 80 s. Whisper translating the audio on this CPU was
too slow (small) or produced nonsense on room audio (base) (all 2026-11-09).

    .venv/bin/python scripts/live_translate.py      # Ctrl+C stops; needs transformers, sentencepiece
    .venv/bin/python scripts/live_translate.py --door https://<box>.<tailnet>.ts.net:18710
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
RATE = 16000  # Qwen3-ASR hears a 48 kHz wav wrong; 16 kHz mono s16 is what it expects
FRAME = RATE // 10  # 100 ms of samples
LOG = Path.home() / ".local/state/live_translate" / time.strftime("%Y-%m-%d.log")
CJK = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


def rms(frame: bytes) -> float:
    samples = array.array("h", frame)
    return math.sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


class Chunker:
    """Groups 100 ms frames into utterances: ends one after `pause` quiet frames, or at `longest`."""

    def __init__(self, threshold: float, pause: int = 7, shortest: int = 5, longest: int = 80):
        self.threshold, self.pause, self.shortest, self.longest = threshold, pause, shortest, longest
        self.frames: list[bytes] = []
        self.voiced = self.quiet = 0

    def feed(self, frame: bytes) -> bytes | None:
        loud = rms(frame) > self.threshold
        if not self.frames and not loud:
            return None
        self.frames.append(frame)
        self.voiced += loud
        self.quiet = 0 if loud else self.quiet + 1
        if self.quiet < self.pause and len(self.frames) < self.longest:
            return None
        clip, voiced = b"".join(self.frames), self.voiced
        self.frames, self.voiced, self.quiet = [], 0, 0
        return clip if voiced >= self.shortest else None


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
        with urllib.request.urlopen(req, timeout=timeout) as r:
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


def caption(clips: queue.Queue, language: str, translate) -> None:
    """The Tiiny writes each clip down; the row prints once its text is translated here."""
    while True:
        pcm = clips.get()
        try:
            dutch = transcribe(pcm, language)
        except (RuntimeError, OSError) as e:
            print(f"\033[31m[Tiiny: {e}]\033[0m", flush=True)
            continue
        if not dutch or CJK.search(dutch):  # noise comes back as Chinese
            continue
        english = translate(dutch)
        for row in columns(dutch, english, shutil.get_terminal_size().columns):
            print(row, flush=True)
        print(flush=True)
        with LOG.open("a") as log:  # the session survives the window
            log.write(f"{time.strftime('%H:%M:%S')} NL {dutch}\n{time.strftime('%H:%M:%S')} EN {english}\n")


def main(language: str, model: str) -> int:
    from transformers import MarianMTModel, MarianTokenizer  # only the live path needs it

    print(f"Loading {model}…", flush=True)
    tokenizer, marian = MarianTokenizer.from_pretrained(model), MarianMTModel.from_pretrained(model)

    def translate(dutch: str) -> str:
        batch = tokenizer([dutch], return_tensors="pt", truncation=True)
        out = marian.generate(**batch, max_new_tokens=256, num_beams=2)
        return tokenizer.decode(out[0], skip_special_tokens=True)

    mic = subprocess.Popen(["pw-record", "--rate", str(RATE), "--channels", "1", "--format", "s16",
                            "--raw", "-"], stdout=subprocess.PIPE)
    assert mic.stdout is not None
    stdout = mic.stdout
    read = lambda: stdout.read(FRAME * 2)
    # the quietest 100 ms of 2 s: people may already be talking when this starts
    floor = min(rms(read()) for _ in range(20))
    chunker = Chunker(threshold=max(floor * 1.3, 150))
    print(f"Listening (noise floor {floor:.0f}, speech above {chunker.threshold:.0f}). Ctrl+C stops.\n")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    clips: queue.Queue = queue.Queue()
    threading.Thread(target=caption, args=(clips, language, translate), daemon=True).start()
    try:
        while frame := read():
            if clip := chunker.feed(frame):
                clips.put(clip)
    except KeyboardInterrupt:
        pass
    finally:
        mic.terminate()
    return 0


def selftest() -> None:
    quiet, loud = array.array("h", [10] * FRAME).tobytes(), array.array("h", [3000] * FRAME).tobytes()
    c = Chunker(threshold=150)
    assert c.feed(quiet) is None and not c.frames, "leading silence is dropped"
    out = [c.feed(f) for f in [loud] * 10 + [quiet] * 7]
    assert out[:-1] == [None] * 16 and len(out[-1] or b"") == 17 * FRAME * 2, "cut after a 0.7 s pause"
    assert [c.feed(f) for f in [loud] * 2 + [quiet] * 7][-1] is None, "a cough is not an utterance"
    assert sum(c.feed(loud) is not None for _ in range(80)) == 1, "a long monologue is cut at 8 s"
    rows = columns("een twee drie vier vijf zes", "one two", 24)
    assert len(rows) == 3 and all("│" in r for r in rows), "Dutch and English share rows"
    assert rows[0].index("│") == rows[2].index("│"), "the divider is aligned"
    assert CJK.search("哎呀，我天哪！") and not CJK.search("Goedemiddag, allemaal.")
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--language", default="nl", help="spoken language hint for the Tiiny (it ignores it today)")
    p.add_argument("--door", help=f"HAI OS owner door URL (default: the line in {DOOR_FILE})")
    p.add_argument("--model", default="Helsinki-NLP/opus-mt-nl-en", help="translation model on this laptop")
    args = p.parse_args()
    if args.selftest:
        selftest()
    else:
        try:
            DOOR = (args.door or DOOR_FILE.read_text()).strip().rstrip("/")
        except FileNotFoundError:
            sys.exit(f"No door: pass --door or write its URL into {DOOR_FILE}")
        sys.exit(main(args.language, args.model))
