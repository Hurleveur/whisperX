#!/usr/bin/env python3
"""Take meeting recordings out of the owner's Solid pod, transcribe them, draft the notes.

Run on the home laptop (see docs/meeting-inbox.md). The browser applet PUTs
`<pod>/meetings/inbox/<id>/meta.json` and the recording named by its `audio`. For each such
folder without a `done`, this downloads the recording, runs `meeting.sh --transcribe-only`
(GPU whisperX, or the Tiiny's ASR through the owner door: never whisper on a CPU), asks the
Tiiny's chat model for a summary, next steps, agreements and lessons, and PUTs
`transcript.txt`, `draft.json` and last an empty `done`. A failed folder is retried next run,
except a recording with no speech, which gets a `failed` file and is left alone.

    python scripts/pod_meetings.py [--once] [--interval 120]
    python scripts/pod_meetings.py --selftest

Needs `~/.config/hai/pod-meetings.json` (keys id, secret, pod, issuer; or pod plus `from`, the
Obsidian Solid sync plugin's data.json, whose client credential is then used) and the door
file `~/.config/hai/door`. The secret and the access token are never printed or logged.
Standard library only: the DPoP proofs are signed with the small P-256 code below, because
the venv carries no crypto package.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_translate  # noqa: E402  (post() and the one-request-at-a-time BOX lock)
import meeting_inbox  # noqa: E402  (DOOR_FILE, running_box_model)
import voices  # noqa: E402  (enrolled voices name the diarization labels)
from analyze_meeting import chunk, read_transcript  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CREDS_FILE = Path.home() / ".config/hai/pod-meetings.json"
CREDS_KEYS = ("id", "secret", "pod")
INBOX = "meetings/inbox/"
LANGUAGES = {"nl": "Dutch", "en": "English", "fr": "French"}
# A transcript piece per request, sized from the window of the model that runs (the door's
# /v1/models lists it as hai_box.window). Pieces are drafted blind to each other and merged
# from their JSON alone, which loses who said what and duplicates items, so the larger the
# piece the better, up to what the model still reads closely.
OUTPUT_TOKENS = 3000  # the reply's cap (max_tokens)
PROMPT_TOKENS = 600  # the system prompt and the participants line, with room to spare
CHARS_PER_TOKEN = 3  # cautious: Dutch and French spend more tokens per character than English
FALLBACK_WINDOW = 8192  # a model that does not say: the smallest window a chat model runs with
# Past this a small model drops details from the middle of a long input. Measured once: a
# 3-hour English meeting (100123 chars) drafted in one piece on the Tiiny's 27B in 243 s with
# 10 quoted items, where two pieces took 802 s and the merge lost the owners (2026-11-25).
# shortcut: one meeting; raise it again once a longer one drafts as well.
PIECE_TOKENS_MAX = 40000


def piece_chars(window: int | None) -> int:
    """The longest transcript piece, in characters, one request to a `window`-token model takes."""
    room = (window if isinstance(window, int) and window > 0 else FALLBACK_WINDOW) - OUTPUT_TOKENS - PROMPT_TOKENS
    return max(2000, min(room, PIECE_TOKENS_MAX) * CHARS_PER_TOKEN)


def even_pieces(text: str, size: int) -> list[str]:
    """As few pieces of at most `size` chars as `chunk` gives, cut as evenly as the lines allow."""
    count = max(1, -(-len(text) // size))
    longest = max((len(line) for line in text.splitlines(keepends=True)), default=0)
    return chunk(text, min(size, -(-len(text) // count) + longest))


SUMMARY_MAX, ITEM_MAX, ITEMS_MAX = 4000, 400, 30

DRAFT_PROMPT = (
    "You write the notes of a meeting from its transcript. The transcript comes from speech"
    " recognition and may contain misheard words and speaker labels on the wrong line: use the"
    " intended meaning, and never invent what was not said. Write in {language}. Reply with one"
    " JSON object and nothing else:"
    ' {{"summary": "a short paragraph on what the meeting was about and decided",'
    ' "next": [ITEM], "agreements": [ITEM], "lessons": [ITEM], "nuggets": ["a line said"]}},'
    ' where each ITEM is {{"text": "the item", "quote": "a few exact words of the transcript it'
    ' rests on, copied letter for letter"}}.'
    " A next step is an action still to do after the meeting; its ITEM also has \"owner\": the"
    " name, as the transcript or the participants line writes it, of the person who will do it,"
    " or \"\" when that is not clear. Something done during the meeting is not a next step. An"
    " agreement is something the people explicitly settled together, not one person's statement."
    " A lesson is an insight the people drew. Leave out facts and claims merely mentioned."
    " A nugget is one of the best things said, at most five: a sharp insight, a striking way to"
    " put something, a line that sums the meeting up; one to three sentences copied exactly"
    " from the transcript, letter for letter, in its language, without its speaker label: never"
    " reworded, shortened, translated or stitched together from two places."
    " Fewer, sure items beat many: a list with nothing certain to say is []."
)
COMBINE_PROMPT = (
    "Below are the notes of consecutive parts of one meeting, each a JSON object. Merge them"
    " into the notes of the whole meeting: one summary paragraph, and each list without"
    " duplicates; a next step starting with a name and a colon keeps them. Write in {language},"
    " except the nuggets: keep at most five, each copied unchanged, letter for letter."
    " Reply with one JSON object and nothing else, with the"
    ' keys "summary" (string), "next", "agreements", "lessons" and "nuggets" (lists of strings).'
)


def log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {msg}", file=sys.stderr, flush=True)


# --- ES256 on P-256, stdlib only (an ephemeral key per run: no constant-time needed) ---

P = 2**256 - 2**224 + 2**192 + 2**96 - 1
N = 0xFFFFFFFF00000000FFFFFFFFFFFFFFFFBCE6FAADA7179E84F3B9CAC2FC632551
B = 0x5AC635D8AA3A93E7B3EBBD55769886BC651D06B0CC53B0F63BCE3C3E27D2604B
G = (0x6B17D1F2E12C4247F8BCE6E563A440F277037D812DEB33A0F4A13945D898C296,
     0x4FE342E2FE1A7F9B8EE7EB4A7C0F9E162BCE33576B315ECECBB6406837BF51F5)


def _add(a, b):
    if a is None or b is None:
        return a or b
    if a[0] == b[0] and (a[1] + b[1]) % P == 0:
        return None
    if a == b:
        m = (3 * a[0] * a[0] - 3) * pow(2 * a[1], -1, P) % P
    else:
        m = (b[1] - a[1]) * pow(b[0] - a[0], -1, P) % P
    x = (m * m - a[0] - b[0]) % P
    return x, (m * (a[0] - x) - a[1]) % P


def _mul(k: int, pt):
    out = None
    while k:
        if k & 1:
            out = _add(out, pt)
        pt = _add(pt, pt)
        k >>= 1
    return out


def b64u(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def b64u_dec(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def new_key() -> tuple[int, tuple[int, int]]:
    d = secrets.randbelow(N - 1) + 1
    return d, _mul(d, G)


def es256_sign(d: int, message: bytes) -> bytes:
    z = int.from_bytes(hashlib.sha256(message).digest(), "big")
    while True:
        k = secrets.randbelow(N - 1) + 1
        r = _mul(k, G)[0] % N
        s = pow(k, -1, N) * (z + r * d) % N
        if r and s:
            return r.to_bytes(32, "big") + s.to_bytes(32, "big")  # JWS: raw r||s


def es256_verify(pub: tuple[int, int], message: bytes, sig: bytes) -> bool:
    if len(sig) != 64:
        return False
    r, s = int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:], "big")
    if not (0 < r < N and 0 < s < N):
        return False
    z = int.from_bytes(hashlib.sha256(message).digest(), "big")
    w = pow(s, -1, N)
    pt = _add(_mul(z * w % N, G), _mul(r * w % N, pub))
    return pt is not None and pt[0] % N == r


def dpop_proof(key, method: str, url: str, token: str = "", nonce: str = "") -> str:
    """A DPoP proof JWT (RFC 9449); `ath` binds it to the access token on resource requests."""
    d, (x, y) = key
    header = {"typ": "dpop+jwt", "alg": "ES256",
              "jwk": {"kty": "EC", "crv": "P-256", "x": b64u(x.to_bytes(32, "big")),
                      "y": b64u(y.to_bytes(32, "big"))}}
    claims = {"htm": method, "htu": url.split("#")[0].split("?")[0],
              "iat": int(time.time()), "jti": secrets.token_urlsafe(16)}
    if token:
        claims["ath"] = b64u(hashlib.sha256(token.encode()).digest())
    if nonce:
        claims["nonce"] = nonce
    signing = b64u(json.dumps(header).encode()) + "." + b64u(json.dumps(claims).encode())
    return signing + "." + b64u(es256_sign(d, signing.encode()))


# --- the pod ---

def load_creds(path: Path) -> dict:
    try:
        raw = json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"no pod credentials: create {path} with the keys "
                         f"{', '.join(CREDS_KEYS)} and issuer (see docs/meeting-inbox.md)")
    except ValueError:
        raise SystemExit(f"{path} is not valid JSON; it needs the keys {', '.join(CREDS_KEYS)}")
    # `from`: a client credential already on this machine, the Obsidian Solid sync plugin's,
    # read here by key so it is never copied into a second file.
    if raw.get("from"):
        src = Path(raw["from"]).expanduser()
        try:
            plugin = json.loads(src.read_text())
        except (OSError, ValueError) as e:
            raise SystemExit(f"{path}: cannot read the credential file it names, {src} ({type(e).__name__})")
        for key, theirs in (("id", "clientId"), ("secret", "clientSecret"), ("issuer", "issuer")):
            raw[key] = raw.get(key) or plugin.get(theirs)
    missing = [k for k in CREDS_KEYS if not isinstance(raw.get(k), str) or not raw[k]]
    if missing:
        raise SystemExit(f"{path} lacks the keys: {', '.join(missing)}")
    if not raw["pod"].endswith("/"):
        raise SystemExit(f"{path}: 'pod' must be the pod root URL ending in /")
    # a Community Solid Server is its own issuer: the pod's origin, unless the file says otherwise
    raw["issuer"] = raw.get("issuer") or urllib.parse.urljoin(raw["pod"], "/")
    if not raw["issuer"].endswith("/"):
        raw["issuer"] += "/"
    return raw


class Pod:
    def __init__(self, creds: dict):
        self.root = creds["pod"]
        self._id, self._secret, self.issuer = creds["id"], creds["secret"], creds["issuer"]
        self.key = new_key()
        self.token, self.expires, self.nonce = "", 0.0, ""

    def _fetch(self, method, url, body=None, headers=None):
        """One HTTP call with a fresh proof; retries once when the server asks for a DPoP nonce."""
        for attempt in (0, 1):
            h = {**(headers or {})}
            bound = h.get("Authorization", "").startswith("DPoP ")  # ath only on pod requests, never the token call
            h["DPoP"] = dpop_proof(self.key, method, url, self.token if bound else "", self.nonce)
            req = urllib.request.Request(url, data=body, headers=h, method=method)
            try:
                return urllib.request.urlopen(req, timeout=120)
            except urllib.error.HTTPError as e:
                nonce = e.headers.get("DPoP-Nonce")
                if nonce and nonce != self.nonce and attempt == 0:
                    self.nonce = nonce
                    continue
                raise

    def _login(self) -> None:
        cfg = json.load(urllib.request.urlopen(f"{self.issuer}.well-known/openid-configuration", timeout=30))
        basic = f"{urllib.parse.quote(self._id, safe='')}:{urllib.parse.quote(self._secret, safe='')}"
        with self._fetch("POST", cfg["token_endpoint"], b"grant_type=client_credentials&scope=webid", {
                "Authorization": "Basic " + base64.b64encode(basic.encode()).decode(),
                "Content-Type": "application/x-www-form-urlencoded"}) as r:
            answer = json.load(r)
        self.token = answer["access_token"]
        self.expires = time.time() + float(answer.get("expires_in", 600)) - 30

    def request(self, method: str, url: str, body: bytes | None = None, ctype: str = "",
                accept: str = ""):
        if time.time() >= self.expires:
            self._login()
        h = {"Authorization": "DPoP " + self.token}
        if ctype:
            h["Content-Type"] = ctype
        if accept:
            h["Accept"] = accept
        try:
            return self._fetch(method, url, body, h)
        except urllib.error.HTTPError as e:
            if e.code != 401:
                raise
            self._login()  # token revoked or expired early: once more with a new one
            h["Authorization"] = "DPoP " + self.token
            return self._fetch(method, url, body, h)

    def get_bytes(self, url: str) -> bytes:
        with self.request("GET", url) as r:
            return r.read()

    def put(self, url: str, body: bytes, ctype: str) -> None:
        with self.request("PUT", url, body, ctype):
            pass

    def children(self, container: str) -> list[str]:
        try:
            with self.request("GET", container, accept="text/turtle") as r:
                return contained(r.read().decode(), container)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return []
            raise


CONTAINS = re.compile(r"(?:ldp:contains|<http://www\.w3\.org/ns/ldp#contains>)((?:\s*<[^>]*>\s*,?)+)")


def contained(turtle: str, base: str) -> list[str]:
    """Member URLs of a container listing: the `ldp:contains` objects, resolved against base."""
    out = []
    for m in CONTAINS.finditer(turtle):
        out += [urllib.parse.urljoin(base, iri) for iri in re.findall(r"<([^>]*)>", m.group(1))]
    return out


def name_of(url: str) -> str:
    return urllib.parse.unquote(url.rstrip("/").rsplit("/", 1)[-1])


def decide(meta: dict | None, names: set[str]) -> str:
    """'process' only for a folder with meta.json and its audio and no done."""
    if "done" in names:
        return "skip: done"
    if "failed" in names:
        return "skip: failed"
    if meta is None or "meta.json" not in names:
        return "skip: no meta.json"
    audio = meta.get("audio")
    if not isinstance(audio, str) or audio not in names:
        return "skip: audio not there yet"
    return "process"


# --- the work on one recording ---

SPEAKER_LABEL = re.compile(r"^[\w-]{1,40}:[ \t]*", re.M)  # read_transcript's "Neil: ", "SPEAKER_01: "
UNNAMED = re.compile(r"SPEAKER_\d+$|UNKNOWN$")  # a voice nobody was named for


def _plain(text: str) -> str:
    """Lowercase words only, so a quote matches the transcript whatever its punctuation."""
    return " ".join(re.findall(r"\w+", text.casefold()))


def coerce_draft(obj, source: str | None = None, participants: tuple = ()) -> dict:
    """What the applet reads: a summary string and four lists of strings, whatever came back.

    With `source`, an item survives only if it carries a quote found in that transcript text:
    the model's own claims and items about nothing said are dropped, not shown as agreed.
    A nugget is its own quote, in one piece: a line reworded or stitched together is dropped.
    """
    obj = obj if isinstance(obj, dict) else {}
    summary = obj.get("summary")
    out = {"summary": summary.strip()[:SUMMARY_MAX] if isinstance(summary, str) else ""}
    # Speaker labels are not words said: a quote over one speaker's next line skips them.
    said, starts = "", []  # starts: (offset in said, the line's speaker name or None)
    for line in (source or "").splitlines():
        label = SPEAKER_LABEL.match(line)
        words = _plain(line[label.end():] if label else line)
        if words:
            said += (" " if said else "") + words
            name = label.group(0).rstrip(": \t") if label else ""
            starts.append((len(said) - len(words), None if UNNAMED.match(name) else name))
    known = [n for n in dict.fromkeys([s for _, s in starts if s] + list(participants)) if n]

    def owner(i: dict, at: int) -> str | None:
        """The model's pick if it is a known person (a short "Alex" is "Alexandre"), else the quote's speaker."""
        pick = i.get("owner").strip().casefold() if isinstance(i.get("owner"), str) else ""
        for n in known if pick else []:
            if n.casefold().startswith(pick) or pick.startswith(n.casefold()):
                return n
        return next((s for o, s in reversed(starts) if o <= at), None)

    for key in ("next", "agreements", "lessons", "nuggets"):
        items = obj.get(key)
        kept = []
        for i in items if isinstance(items, list) else []:
            text = i.get("text") if isinstance(i, dict) else i if source is None or key == "nuggets" else None
            if not isinstance(text, str) or not text.strip():
                continue
            if source is not None:
                if key == "nuggets":
                    text = SPEAKER_LABEL.sub("", text.strip())
                    parts = [_plain(text)]
                else:
                    quote = i.get("quote") if isinstance(i.get("quote"), str) else ""
                    parts = [_plain(SPEAKER_LABEL.sub("", p.strip())) for p in re.split(r"\.\.\.|…", quote)]
                parts = [p for p in parts if p]
                if sum(map(len, parts)) < 8 or not all(p in said for p in parts):
                    continue
                who = owner(i, said.find(parts[0])) if key == "next" else None
                if who and not text.strip().casefold().startswith(who.casefold()):
                    text = f"{who}: {text.strip()}"
            kept.append(text.strip()[:ITEM_MAX])
        out[key] = kept[:ITEMS_MAX]
    return out


def extract_json(reply: str):
    """The JSON object in a model's reply, which may wrap it in prose or a code fence."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the model's reply holds no JSON object")
    return json.loads(reply[start:end + 1])


def ask(door: str, model: str, system: str, user: str) -> dict:
    live_translate.DOOR = door
    body = json.dumps({"model": model, "temperature": 0.2, "max_tokens": OUTPUT_TOKENS,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    # The door has no read timeout and queues a call behind the box's other work: one
    # 120000-char piece took 243 s alone and over 600 s on a shared Tiiny (2026-11-26).
    answer = live_translate.post("/v1/chat/completions", body, "application/json", timeout=1800)
    if "error" in answer:
        raise RuntimeError(answer["error"].get("message", "the door refused the chat request"))
    return extract_json(answer["choices"][0]["message"]["content"])


TRIES = 3  # the box model's JSON is sometimes broken; one bad piece must not throw the others away


def draft_notes(transcript: str, language: str, door: str, model: str, ask_fn=ask,
                participants: list | None = None, window: int | None = None) -> dict:
    lang = LANGUAGES.get(language, "the language the meeting was held in")
    system = DRAFT_PROMPT.format(language=lang)

    people = [p.strip() for p in participants or [] if isinstance(p, str) and p.strip()]
    lead = f"Participants: {', '.join(people)}.\n\nTranscript:\n" if people else ""

    def asked(system: str, user: str, source: str | None) -> dict:
        for left in range(TRIES - 1, -1, -1):
            try:
                return coerce_draft(ask_fn(door, model, system, user), source, people)
            except ValueError:  # json.JSONDecodeError included: asked again, the same piece alone
                if not left:
                    raise

    parts = [asked(system, lead + c, c) for c in even_pieces(transcript, piece_chars(window))]
    if len(parts) == 1:
        return parts[0]
    if not parts:
        return coerce_draft({})
    # The pieces' items are already checked against their own text; the merge only rewords them.
    return asked(COMBINE_PROMPT.format(language=lang),
                 "\n\n".join(json.dumps(p, ensure_ascii=False) for p in parts), None)


def asr_name() -> str:
    """The ASR meeting.sh will use, decided the way it decides; refuses whisper without a GPU."""
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0 \
        if shutil.which("nvidia-smi") else False
    asr = os.environ.get("ASR") or ("whisperx" if gpu else "tiiny")
    if asr == "whisperx" and not gpu:
        raise RuntimeError("ASR=whisperx but nvidia-smi sees no GPU: whisper never runs on a CPU here")
    return asr


MIN_SHARE = 0.05  # a label with fewer of the lines is a stray split of diarization, not a person
# A participant's enrolled voice through a phone scored 0.48-0.59 in five meetings (2026-11-23,
# laptop-enrolled), every other voice 0.30 at most: closest by this much is that person.
NEAR, MARGIN = 0.45, 0.2


def same_person(a: str, b: str) -> bool:
    """Two case-folded names one person goes by: equal, one the short form of the other
    ("alex", "alexandre"), or the same first name ("alexandre de groodt", "alexandre")."""
    a, b = a.split(), b.split()
    if not a or not b:
        return False
    return a[0] == b[0] or (len(a[0]) >= 3 and len(b[0]) >= 3 and (a[0].startswith(b[0]) or b[0].startswith(a[0])))


def speaker_names(emb: dict, known: dict, floor: float, participants: list,
                  lines: dict | None = None) -> tuple[dict[str, str], dict[str, str]]:
    """({label: name}, {label: name} learned). The voices enrolled on this box first
    (voices.match); then, when exactly one label of any weight (MIN_SHARE of the lines) and
    exactly one of the meeting's participants are left, that label is that person, and is
    returned as learned so its voice can be enrolled. Any other leftover keeps its SPEAKER_NN:
    a wrong name in the notes, or in the voices, is worse than none.
    Between the two, a participant whose voice is enrolled but heard through another
    microphone (under `floor`) is the one real label clearly closest to it (NEAR, MARGIN
    ahead of the next), and is learned too, so that microphone's voice is known next time."""
    named = voices.match(emb, known, floor)
    total = sum((lines or {}).values())
    real = [lab for lab in emb if not total or (lines or {}).get(lab, 0) / total >= MIN_SHARE]
    people = {p.strip().casefold(): p.strip() for p in reversed(participants) if isinstance(p, str) and p.strip()}
    learned = {}
    for k in people:
        for name, samples in known.items():
            if not samples or name in named.values() or not same_person(k, name.casefold()):
                continue
            near = sorted(((max(voices.cosine(emb[lab], s) for s in samples), lab) for lab in real if lab not in named), reverse=True)
            if near and near[0][0] >= NEAR and (len(near) == 1 or near[0][0] - near[1][0] >= MARGIN):
                named[near[0][1]] = learned[near[0][1]] = name
    taken = {n.casefold() for n in named.values()}
    left_people = [p for k, p in people.items() if not any(same_person(k, t) for t in taken)]
    left_labels = [lab for lab in real if lab not in named]
    if len(left_labels) == 1 and len(left_people) == 1:
        learned[left_labels[0]] = left_people[0]
    return {**named, **learned}, learned


def transcribe(audio: Path, language: str, participants: list | None = None) -> str:
    sh = os.environ.get("MEETING_SH") or str(REPO / "scripts/meeting.sh")  # the override is for tests
    asr = asr_name()
    cmd = [sh, "--transcribe-only", str(audio)]
    # Any other language (HyperMeet's "mix", or none) is detected per speech chunk among the
    # three, so a meeting that switches is not written all in the language of its first 30 s.
    if asr == "whisperx":
        cmd += ["--language", language] if language in LANGUAGES else ["--languages", ",".join(LANGUAGES)]
    run = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "ASR": asr})
    if run.returncode:
        raise RuntimeError(f"meeting.sh failed ({run.returncode}): {run.stderr.strip()[-300:]}")
    lines = run.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("meeting.sh printed no transcript path")
    path = Path(lines[-1])
    # --transcribe-only stops before meeting.sh names the voices, so the names are put here.
    names = {}
    if path.suffix == ".json":
        store = voices.voices_dir()
        segments = json.loads(path.read_text()).get("segments") or []
        lines = {}
        for seg in segments:
            if seg.get("speaker"):
                lines[seg["speaker"]] = lines.get(seg["speaker"], 0) + 1
        names, learned = speaker_names(voices.embeddings(path), voices.load(store),
                                       voices.threshold(), participants or [], lines)  # fmt: skip
        log(f"speakers named: {', '.join(f'{k}={v}' for k, v in sorted(names.items())) or 'none'}")
        # Told by the participant list rather than heard: kept, so the next meeting knows the voice.
        for label, name in learned.items():
            try:
                n = voices.enroll(path, label, name, store)
                log(f"voice learned: {name} ({n} sample(s))")
            except SystemExit as e:  # an unusable name or no embedding: the transcript still goes out
                log(f"voice not learned for {name}: {e}")
    return read_transcript(path, names)


def write_results(put, folder: str, transcript: str, draft: dict) -> None:
    """The result files, `done` last so a reader never sees half of them."""
    put(folder + "transcript.txt", transcript.encode(), "text/plain")
    put(folder + "draft.json", json.dumps(draft, ensure_ascii=False, indent=2).encode(), "application/json")
    put(folder + "done", b"", "text/plain")


def meeting_parts(folders: list[str], mid: str) -> list[str]:
    """One meeting's inbox folders in the order sent: <id>, then <id>-2, <id>-3 (a paused
    recording goes up one part per folder)."""
    mine = [u for u in folders if re.fullmatch(re.escape(mid) + r"(-\d+)?", name_of(u))]
    return sorted(mine, key=lambda u: int(name_of(u)[len(mid) + 1:] or 1))


def whole_transcript(children, get_bytes, parts: list[str], folder: str, transcript: str) -> str | None:
    """The meeting's transcript, every part in order; None while another part is still to be
    heard, since the last part heard drafts the whole meeting. A part the phone already took
    into a note has no transcript.txt left and is missed."""
    said = []
    for p in parts:
        if p == folder:
            said.append(transcript)
            continue
        names = {name_of(u) for u in children(p)}
        if "failed" in names:
            continue
        if "done" not in names:
            return None
        said.append(get_bytes(p + "transcript.txt").decode() if "transcript.txt" in names else "")
    return "\n\n".join(s.strip() for s in said if s.strip())


def process_folder(pod: Pod, folder: str, door: str, model: str, window: int | None = None,
                   folders: list[str] = ()) -> str:
    """Returns what happened; raises on a failure so the folder is retried next run."""
    names = {name_of(u) for u in pod.children(folder)}
    meta = None
    if "meta.json" in names:
        meta = json.loads(pod.get_bytes(folder + "meta.json"))
    verdict = decide(meta, names)
    if verdict != "process":
        return verdict
    # meeting.sh keeps transcripts/<audio stem>.json and skips a stem it has seen: the folder
    # name in front keeps two recordings both called rec.webm apart. A pod name never picks a path.
    audio_name = f"{name_of(folder)}-{Path(meta['audio']).name}"
    language = meta.get("language") if isinstance(meta.get("language"), str) else ""
    with tempfile.TemporaryDirectory(prefix="pod-meetings-") as tmp:
        audio = Path(tmp) / audio_name
        with pod.request("GET", folder + urllib.parse.quote(meta["audio"])) as r, audio.open("wb") as f:
            shutil.copyfileobj(r, f)
        people = meta.get("participants") if isinstance(meta.get("participants"), list) else []
        transcript = transcribe(audio, language, people)
    mid = meta.get("id") if isinstance(meta.get("id"), str) else ""
    parts = meeting_parts(list(folders), mid) if mid else []
    parts = parts if folder in parts else [folder]
    whole = whole_transcript(pod.children, pod.get_bytes, parts, folder, transcript)
    if not transcript.strip() and not whole:
        # meeting.sh caches the empty result under the audio stem, so a retry can never do
        # better: mark the folder for good instead of downloading it again every pass.
        pod.put(folder + "failed", b"no speech in the recording\n", "text/plain")
        return "failed: no speech in the recording"
    if whole is None:
        # An empty draft: the phone shows nothing for this part, and the part heard last
        # brings the whole meeting's.
        write_results(pod.put, folder, transcript, coerce_draft({}))
        return "done, drafted with the meeting's last part"
    draft = draft_notes(whole, language, door, model, participants=people, window=window)
    # The phone drops the proposals of every part named here: one meeting, one draft.
    draft["parts"] = [name_of(p) for p in parts]
    write_results(pod.put, folder, transcript, draft)
    return "done" if len(parts) == 1 else f"done, drafted the whole meeting ({len(parts)} parts)"


def box_model(door: str, listing: dict | None = None) -> tuple[str, int | None]:
    """The chat model running on the owner's box, and its window in tokens when the box says."""
    if listing is None:
        with urllib.request.urlopen(f"{door}/v1/models", timeout=10) as resp:
            listing = json.load(resp)
    model = meeting_inbox.running_box_model(door, listing)
    rows = [m for m in listing.get("data", []) if m.get("id") == model]
    window = (rows[0].get("hai_box") or {}).get("window") if rows else None
    return model, window if isinstance(window, int) else None


def run_once(pod: Pod, door: str) -> int:
    """One pass over the inbox; the number of folders that failed."""
    failed = 0
    try:
        folders = [u for u in pod.children(pod.root + INBOX) if u.endswith("/")]
        model, window = box_model(door) if folders else ("", None)
    except (urllib.error.URLError, OSError, KeyError, ValueError, SystemExit) as e:
        log(f"inbox not read: {type(e).__name__}: {e}")
        return 1
    for folder in folders:
        try:
            verdict = process_folder(pod, folder, door, model, window, folders)
            log(f"{name_of(folder)}: {verdict}")
        except Exception as e:  # noqa: BLE001  one folder's failure must not stop the others
            failed += 1
            log(f"{name_of(folder)}: FAILED, retried next run: {type(e).__name__}: {e}")
    return failed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--once", action="store_true", help="one pass, then exit (1 if a folder failed)")
    ap.add_argument("--interval", type=int, default=120, help="seconds between passes (default 120)")
    ap.add_argument("--door", help="owner door URL, instead of the door file (tests)")
    ap.add_argument("--creds", type=Path, default=CREDS_FILE, help="credentials file (default %(default)s)")
    args = ap.parse_args()
    creds = load_creds(args.creds)
    try:
        door = args.door or meeting_inbox.DOOR_FILE.read_text().strip()
    except FileNotFoundError:
        raise SystemExit(f"no door: put the owner door's https URL on one line in {meeting_inbox.DOOR_FILE}")
    door = door.rstrip("/")
    pod = Pod(creds)
    while True:
        failed = run_once(pod, door)
        if args.once:
            return 1 if failed else 0
        time.sleep(args.interval)


def selftest() -> None:
    # CSS container listing, real shape: prefixes, a type list, relative members, child metadata.
    base = "https://pod.example/alex/meetings/inbox/"
    listing = """@prefix dc: <http://purl.org/dc/terms/>.
@prefix ldp: <http://www.w3.org/ns/ldp#>.
@prefix posix: <http://www.w3.org/ns/posix/stat#>.
@prefix xsd: <http://www.w3.org/2001/XMLSchema#>.

<> a ldp:Container, ldp:BasicContainer, ldp:Resource;
    dc:modified "2026-11-09T10:00:00.000Z"^^xsd:dateTime;
    ldp:contains <2026-11-09-sync/>, <m%201/>, <other.txt>.
<2026-11-09-sync/> a ldp:Container, ldp:BasicContainer, ldp:Resource;
    dc:modified "2026-11-09T10:00:00.000Z"^^xsd:dateTime.
<other.txt> a ldp:Resource; posix:size 12 .
"""
    got = contained(listing, base)
    assert got == [base + "2026-11-09-sync/", base + "m%201/", base + "other.txt"], got
    assert [name_of(u) for u in got] == ["2026-11-09-sync", "m 1", "other.txt"]
    assert contained("<> a ldp:Container.", base) == []
    full = "<> <http://www.w3.org/ns/ldp#contains> <a>,\n <b/> ."
    assert contained(full, base) == [base + "a", base + "b/"]

    # DPoP proof: shape, and the signature verifies against the jwk in its own header.
    key = new_key()
    assert _mul(N, G) is None and (G[1] ** 2 - G[0] ** 3 + 3 * G[0] - B) % P == 0
    url = "https://pod.example/alex/a b?x=1#f"
    proof = dpop_proof(key, "PUT", url, "tok", "n1")
    h64, c64, s64 = proof.split(".")
    header, claims = json.loads(b64u_dec(h64)), json.loads(b64u_dec(c64))
    assert header["typ"] == "dpop+jwt" and header["alg"] == "ES256", header
    assert header["jwk"]["kty"] == "EC" and header["jwk"]["crv"] == "P-256" and "d" not in header["jwk"]
    assert claims["htm"] == "PUT" and claims["htu"] == "https://pod.example/alex/a b", claims
    assert abs(claims["iat"] - time.time()) < 5 and claims["jti"] and claims["nonce"] == "n1"
    assert claims["ath"] == b64u(hashlib.sha256(b"tok").digest())
    assert "ath" not in json.loads(b64u_dec(dpop_proof(key, "POST", url).split(".")[1]))
    pub = (int.from_bytes(b64u_dec(header["jwk"]["x"]), "big"), int.from_bytes(b64u_dec(header["jwk"]["y"]), "big"))
    assert pub == key[1] and es256_verify(pub, f"{h64}.{c64}".encode(), b64u_dec(s64))
    assert not es256_verify(pub, f"{h64}.{c64}x".encode(), b64u_dec(s64)), "a changed message must fail"
    assert not es256_verify(new_key()[1], f"{h64}.{c64}".encode(), b64u_dec(s64)), "another key must fail"
    assert dpop_proof(key, "GET", url) != dpop_proof(key, "GET", url), "fresh jti each time"

    # Draft coercion.
    assert coerce_draft(None) == coerce_draft("x") == {"summary": "", "next": [], "agreements": [], "lessons": [], "nuggets": []}
    d = coerce_draft({"summary": " ok ", "next": ["a", 3, None, "  ", " b "], "agreements": "no",
                      "lessons": ["x" * 999] * 40, "extra": 1})
    assert d["summary"] == "ok" and d["next"] == ["a", "b"] and d["agreements"] == []
    assert len(d["lessons"]) == ITEMS_MAX and len(d["lessons"][0]) == ITEM_MAX and set(d) == {
        "summary", "next", "agreements", "lessons", "nuggets"}
    assert len(coerce_draft({"summary": "s" * 9999})["summary"]) == SUMMARY_MAX
    assert extract_json('Here:\n```json\n{"summary": "x"}\n```') == {"summary": "x"}
    # A long transcript is asked in pieces, then combined once.
    calls = []
    fake = lambda door, model, system, user: (calls.append((system, user)), {"summary": "s", "next": [user[:1]]})[1]
    draft_notes("a\n" * piece_chars(None), "fr", "d", "m", fake)
    assert len(calls) == 3 and "French" in calls[0][0] and calls[-1][0].startswith("Below are"), len(calls)
    calls.clear()
    draft_notes("short", "nl", "d", "m", fake)
    assert len(calls) == 1 and "Dutch" in calls[0][0]
    # The piece follows the running model's window: one request where the window holds the
    # meeting, several where it does not, a cautious size when the model does not say.
    meeting = "Neil: something said in the meeting\n" * 600  # about 21k chars, ~28 minutes
    for window, pieces in ((131072, 1), (32768, 1), (8192, 2), (None, 2), (0, 2)):
        calls.clear()
        draft_notes(meeting, "en", "d", "m", fake, window=window)
        assert len(calls) == pieces + (pieces > 1), (window, len(calls))
    assert piece_chars(10**7) == PIECE_TOKENS_MAX * CHARS_PER_TOKEN and piece_chars(4000) == 2000
    # A meeting just over one piece is cut in even halves, never a full piece and a short tail
    # (2026-11-23: 98268 + 1855 chars, the tail's notes merged blind into the rest).
    line = "Neil: something said in the meeting\n"
    over = line * (piece_chars(None) // len(line) + 2)
    sizes = [len(p) for p in even_pieces(over, piece_chars(None))]
    assert len(sizes) == 2 and abs(sizes[0] - sizes[1]) <= 2 * len(line), sizes
    assert even_pieces("", 10) == [] and "".join(even_pieces(meeting, 5000)) == meeting
    listing = {"data": [{"id": "box/t/a", "hai_box": {"loaded": False, "window": 4096}},
                        {"id": "box/t/b", "hai_box": {"loaded": True, "window": 131072}}]}
    assert box_model("d", listing) == ("box/t/b", 131072)
    listing["data"][1]["hai_box"].pop("window")
    assert box_model("d", listing) == ("box/t/b", None)
    # An item stands only on a quote of the transcript; a bare string or an invented quote is dropped.
    said = "Neil: We'll come to that.\nAlexandre: I should make something for a timer in here, maybe."
    reply = {"summary": "s", "next": [
        {"text": "Alexandre could add a timer", "quote": "I should make something for a TIMER in here"},
        {"text": "invented", "quote": "the .nc domain was cancelled in 1993"},
        {"text": "too short", "quote": "a"}, "a bare string"]}
    calls.clear()
    d = draft_notes(said, "en", "d", "m", lambda *a: (calls.append(a[3]), reply)[1], participants=["Alex", " Neil"])
    assert d["next"] == ["Alexandre could add a timer"], d
    assert calls[0].startswith("Participants: Alex, Neil.\n\nTranscript:\nNeil:"), calls[0][:60]
    # A quote may run over one speaker's consecutive lines, or skip words with "...": its
    # parts are each found once the speaker labels are out (2026-11-23: 9 of 10 items dropped).
    said = "Neil: So the chapter definition.\nNeil: I'm defining the elements.\nSPEAKER_01: Yes, the red flag."
    reply = {"summary": "s", "agreements": [
        {"text": "over two lines", "quote": "So the chapter definition. I'm defining the elements."},
        {"text": "with a gap", "quote": "So the chapter definition ... the red flag"},
        {"text": "one part invented", "quote": "So the chapter definition … the green flag"},
        {"text": "label as a word", "quote": "definition neil i m defining"}]}
    d = draft_notes(said, "en", "d", "m", lambda *a: reply)
    assert d["agreements"] == ["over two lines", "with a gap"], d
    # A next step names who does it: the model's pick when it is a known person, else the
    # named speaker of its quote; an unnamed voice or an unknown name gives no owner.
    said = ("Neil: I know what I need to do next, generate the menu.\n"
            "Neil: Alexandre, can you send the roadmap to me?\nAlexandre: Sure.\n"
            "SPEAKER_02: I will book the room.")
    reply = {"summary": "s", "next": [
        {"text": "Generate the menu", "quote": "what I need to do next, generate the menu"},
        {"text": "Send the roadmap", "owner": "Alex", "quote": "can you send the roadmap"},
        {"text": "Book the room", "quote": "I will book the room"},
        {"text": "Neil: Review it", "owner": "Bob", "quote": "generate the menu"}]}
    d = draft_notes(said, "en", "d", "m", lambda *a: reply, participants=["Alex", "Neil"])
    assert d["next"] == ["Neil: Generate the menu", "Alexandre: Send the roadmap", "Book the room",
                         "Neil: Review it"], d
    # A quote copied with its line's label, as the model often does, is still found.
    reply = {"summary": "s", "agreements": [
        {"text": "Roadmap", "quote": "Neil: Alexandre, can you send ... Alexandre: Sure."}]}
    assert draft_notes(said, "en", "d", "m", lambda *a: reply)["agreements"] == ["Roadmap"]
    # A nugget stands only as words said, whole: reworded, stitched with "..." or too short is dropped.
    reply = {"summary": "s", "nuggets": [
        "I know what I need to do next, generate the menu.", "I know what I need to do, generate the menu.",
        "I know what I need ... the menu", "Neil: Alexandre, can you send the roadmap to me?", "Sure.",
        {"text": "Alexandre: Sure. SPEAKER_02: I will book", "quote": "x"}]}
    assert draft_notes(said, "en", "d", "m", lambda *a: reply)["nuggets"] == [
        "I know what I need to do next, generate the menu.", "Alexandre, can you send the roadmap to me?",
    ], "a leading label goes; one inside is two lines stitched together"
    # The merge of pieces is not checked again: its input is JSON, not the transcript.
    assert coerce_draft({"next": ["merged", {"text": "obj"}]})["next"] == ["merged", "obj"]
    # A piece whose reply is broken JSON is asked again alone, up to TRIES times, then fails.
    calls.clear()
    broken = [0]
    def flaky(door, model, system, user):
        calls.append(user)
        if user.startswith("a") and broken[0] < TRIES - 1:  # every piece fails all but its last try
            broken[0] += 1
            return json.loads("{broken")
        broken[0] = 0
        return {"summary": "s"}
    assert draft_notes("a\n" * piece_chars(None), "fr", "d", "m", flaky)["summary"] == "s"
    assert len(calls) == 2 * TRIES + 1, len(calls)
    try:
        draft_notes("short", "nl", "d", "m", lambda *a: json.loads("{"))
        raise AssertionError("a piece broken every time must fail the draft")
    except ValueError:
        pass

    # The state decision per folder.
    meta = {"audio": "rec.webm"}
    assert decide(meta, {"meta.json", "rec.webm"}) == "process"
    assert decide(meta, {"meta.json", "rec.webm", "transcript.txt"}) == "process", "a half result is redone"
    assert decide(meta, {"meta.json", "rec.webm", "done"}).startswith("skip")
    assert decide(None, {"rec.webm"}).startswith("skip") and decide(meta, {"rec.webm"}).startswith("skip")
    assert decide(meta, {"meta.json"}).startswith("skip"), "audio still uploading"
    assert decide({"audio": 5}, {"meta.json"}).startswith("skip")
    assert decide(meta, {"meta.json", "rec.webm", "failed"}) == "skip: failed"

    # A paused meeting: its folders in the order sent, then one transcript once every part is heard.
    ib = "http://p/inbox/"
    listed = [ib + n + "/" for n in ("m-10", "m", "mx", "m-2", "other", "m-x")]
    assert meeting_parts(listed, "m") == [ib + "m/", ib + "m-2/", ib + "m-10/"], meeting_parts(listed, "m")
    parts = [ib + "m/", ib + "m-2/", ib + "m-3/"]
    files = {ib + "m/": {"done": b"", "transcript.txt": b"Neil: one\n"}, ib + "m-2/": {"meta.json": b"{}"}}
    kids = lambda p: [p + n for n in files.get(p, {})]
    got = lambda u: files[u.rsplit("/", 1)[0] + "/"][u.rsplit("/", 1)[1]]
    assert whole_transcript(kids, got, parts, ib + "m-3/", "Alex: three") is None, "part 2 still to be heard"
    files[ib + "m-2/"] = {"failed": b""}
    assert whole_transcript(kids, got, parts, ib + "m-3/", "Alex: three") == "Neil: one\n\nAlex: three"
    files[ib + "m-2/"] = {"done": b"", "transcript.txt": b"Nico: two"}
    assert whole_transcript(kids, got, parts, ib + "m-3/", "Alex: three") == "Neil: one\n\nNico: two\n\nAlex: three"
    assert whole_transcript(kids, got, parts[:2], ib + "m/", "Neil: again") == "Neil: again\n\nNico: two"
    assert whole_transcript(kids, got, parts, ib + "m/", "Neil: again") is None, "m-3 listed, not uploaded yet"
    files[ib + "m/"] = {"done": b""}
    assert whole_transcript(kids, got, parts[:2], ib + "m-2/", "") == "", "a part already taken by the phone is missed"

    # Write order: done last, with the content types the applet reads.
    puts = []
    write_results(lambda u, b, c: puts.append((u, b, c)), "http://p/f/", "words", {"summary": "s"})
    assert [p[0] for p in puts] == ["http://p/f/transcript.txt", "http://p/f/draft.json", "http://p/f/done"]
    assert [p[2] for p in puts] == ["text/plain", "application/json", "text/plain"] and puts[2][1] == b""
    assert json.loads(puts[1][1]) == {"summary": "s"}

    # Credentials file: a clear exit, never a traceback, never the secret.
    tmp = Path(tempfile.mkdtemp())
    for content, needle in [(None, str(tmp / "c.json")), ("{", "not valid JSON"),
                            ('{"id": "i", "secret": "SECRETVALUE"}', "lacks the keys: pod"),
                            ('{"from": "%s", "pod": "https://x/y/"}' % (tmp / "nope.json"), "cannot read")]:
        if content is not None:
            (tmp / "c.json").write_text(content)
        try:
            load_creds(tmp / "c.json")
            raise AssertionError("accepted bad credentials")
        except SystemExit as e:
            assert needle in str(e) and "SECRETVALUE" not in str(e), e
    (tmp / "c.json").write_text('{"id":"i","secret":"s","webid":"w","pod":"https://x/y/"}')
    assert load_creds(tmp / "c.json")["issuer"] == "https://x/"
    (tmp / "p.json").write_text('{"clientId":"pi","clientSecret":"ps","issuer":"https://idp.example"}')
    (tmp / "c.json").write_text(json.dumps({"from": str(tmp / "p.json"), "pod": "https://x/y/"}))
    got = load_creds(tmp / "c.json")
    assert (got["id"], got["secret"], got["issuer"]) == ("pi", "ps", "https://idp.example/"), got
    shutil.rmtree(tmp)

    # The language the phone sent: one of the three is set, anything else ("mix") is detected
    # per chunk among them. A stand-in meeting.sh records what it was asked.
    tmp = Path(tempfile.mkdtemp())
    (tmp / "sh").write_text('#!/bin/sh\necho "$@" > "%s/args"\necho "Neil: hi" > "%s/t.txt"\necho "%s/t.txt"\n' % (tmp, tmp, tmp))
    (tmp / "sh").chmod(0o755)
    was, os.environ["MEETING_SH"] = os.environ.get("MEETING_SH"), str(tmp / "sh")
    real, globals()["asr_name"] = asr_name, lambda: "whisperx"
    try:
        for language, want in [("fr", "--language fr"), ("mix", "--languages nl,en,fr"), ("", "--languages nl,en,fr")]:
            transcribe(tmp / "a.webm", language)
            assert (tmp / "args").read_text().split(str(tmp / "a.webm"))[1].strip() == want, (language, (tmp / "args").read_text())
    finally:
        globals()["asr_name"] = real
        os.environ.pop("MEETING_SH") if was is None else os.environ.__setitem__("MEETING_SH", was)
        shutil.rmtree(tmp)

    # Speakers: an enrolled voice names its label; one label and one participant left pair up,
    # and that pairing is what gets learned. A stray label (under MIN_SHARE of the lines) does not count.
    me, other, third = [1.0, 0.0], [0.0, 1.0], [0.6, 0.8]
    known = {"Alex": [me]}
    sn = speaker_names
    assert sn({"S0": me, "S1": other}, known, 0.7, ["Alex", "Neil"]) == ({"S0": "Alex", "S1": "Neil"}, {"S1": "Neil"})
    assert sn({"S0": me, "S1": other}, known, 0.7, ["alex ", " Neil", "Neil"])[0] == {"S0": "Alex", "S1": "Neil"}
    assert sn({"S0": other, "S1": me, "S2": third}, known, 0.7, ["Alex", "Neil"],
              {"S0": 3, "S1": 127, "S2": 102}) == ({"S1": "Alex", "S2": "Neil"}, {"S2": "Neil"})
    # Two real labels or two people left: no guess, nothing learned.
    assert sn({"S0": me, "S1": other, "S2": third}, known, 0.7, ["Alex", "Neil"]) == ({"S0": "Alex"}, {})
    assert sn({"S0": me, "S1": other}, known, 0.7, ["Alex", "Neil", "Ann"]) == ({"S0": "Alex"}, {})
    assert sn({"S0": me, "S1": other}, {}, 0.7, ["Alex", "Neil"]) == ({}, {})
    assert sn({"S0": me}, {}, 0.7, ["Alex"]) == ({"S0": "Alex"}, {"S0": "Alex"})
    # A participant written short or long is the enrolled voice: "Alex" is "Alexandre", so Neil is left.
    assert sn({"S0": me, "S1": other}, {"Alexandre": [me]}, 0.7, ["Alex", "Neil"]) == ({"S0": "Alexandre", "S1": "Neil"}, {"S1": "Neil"})
    assert sn({"S0": me, "S1": other}, {"Alex": [me]}, 0.7, ["Alexandre De Groodt", "Neil"])[1] == {"S1": "Neil"}
    assert not same_person("al", "alexandre") and not same_person("neil", "nicolas")
    # Another microphone: the participant's voice at 0.55, under the floor, the other at 0. It is
    # still that participant (learned, for that microphone), and Neil follows by elimination.
    me3, phone3, other3, half3 = [1.0, 0.0, 0.0], [0.55, 0.835, 0.0], [0.0, 0.0, 1.0], [0.4, 0.0, 0.917]
    assert sn({"S0": phone3, "S1": other3}, {"Alexandre": [me3]}, 0.7, ["Alex", "Neil"]) == (
        {"S0": "Alexandre", "S1": "Neil"}, {"S0": "Alexandre", "S1": "Neil"})
    # Not clearly the closest (0.55 against 0.40), or not near at all: no name, nothing learned.
    assert sn({"S0": phone3, "S1": half3}, {"Alexandre": [me3]}, 0.7, ["Alex", "Neil"]) == ({}, {})
    assert sn({"S0": other3, "S1": [0.3, 0.0, 0.954]}, {"Alexandre": [me3]}, 0.7, ["Alex", "Neil"]) == ({}, {})
    # Not a participant of this meeting: an enrolled voice is never placed by closeness.
    assert sn({"S0": phone3, "S1": other3}, {"Alexandre": [me3]}, 0.7, ["Neil", "Ann"]) == ({}, {})
    # Everyone already known: nothing learned twice.
    assert sn({"S0": me, "S1": other}, {"Alex": [me], "Neil": [other]}, 0.7, ["Alex", "Neil"])[1] == {}
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else sys.exit(main())
