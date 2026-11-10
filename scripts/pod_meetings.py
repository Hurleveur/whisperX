#!/usr/bin/env python3
"""Take meeting recordings out of the owner's Solid pod, transcribe them, draft the notes.

Run on the home laptop (see docs/meeting-inbox.md). The browser applet PUTs
`<pod>/meetings/inbox/<id>/meta.json` and the recording named by its `audio`. For each such
folder without a `done`, this downloads the recording, runs `meeting.sh --transcribe-only`
(GPU whisperX, or the Tiiny's ASR through the owner door: never whisper on a CPU), asks the
Tiiny's chat model for a summary, next steps, agreements and lessons, and PUTs
`transcript.txt`, `draft.json` and last an empty `done`. A failed folder is retried next run.

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
from analyze_meeting import chunk, read_transcript  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
CREDS_FILE = Path.home() / ".config/hai/pod-meetings.json"
CREDS_KEYS = ("id", "secret", "pod")
INBOX = "meetings/inbox/"
LANGUAGES = {"nl": "Dutch", "en": "English", "fr": "French"}
CHUNK_CHARS = 12000  # a transcript piece per request; the Tiiny's context is the limit
SUMMARY_MAX, ITEM_MAX, ITEMS_MAX = 4000, 400, 30

DRAFT_PROMPT = (
    "You write the notes of a meeting from its transcript. The transcript comes from speech"
    " recognition and may contain misheard words: use the intended meaning, and never invent"
    " what was not said. Write in {language}. Reply with one JSON object and nothing else:"
    ' {{"summary": "a short paragraph on what the meeting was about and decided",'
    ' "next": ["each next step, with who does it when the transcript says so"],'
    ' "agreements": ["each thing the people agreed on"],'
    ' "lessons": ["each lesson learned"]}}. A list with nothing to say is [].'
)
COMBINE_PROMPT = (
    "Below are the notes of consecutive parts of one meeting, each a JSON object. Merge them"
    " into the notes of the whole meeting: one summary paragraph, and each list without"
    " duplicates. Write in {language}. Reply with one JSON object and nothing else, with the"
    ' keys "summary" (string), "next", "agreements" and "lessons" (lists of strings).'
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
    if meta is None or "meta.json" not in names:
        return "skip: no meta.json"
    audio = meta.get("audio")
    if not isinstance(audio, str) or audio not in names:
        return "skip: audio not there yet"
    return "process"


# --- the work on one recording ---

def coerce_draft(obj) -> dict:
    """What the applet reads: a summary string and three lists of strings, whatever came back."""
    obj = obj if isinstance(obj, dict) else {}
    summary = obj.get("summary")
    out = {"summary": summary.strip()[:SUMMARY_MAX] if isinstance(summary, str) else ""}
    for key in ("next", "agreements", "lessons"):
        items = obj.get(key)
        items = items if isinstance(items, list) else []
        out[key] = [i.strip()[:ITEM_MAX] for i in items if isinstance(i, str) and i.strip()][:ITEMS_MAX]
    return out


def extract_json(reply: str):
    """The JSON object in a model's reply, which may wrap it in prose or a code fence."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end < start:
        raise ValueError("the model's reply holds no JSON object")
    return json.loads(reply[start:end + 1])


def ask(door: str, model: str, system: str, user: str) -> dict:
    live_translate.DOOR = door
    body = json.dumps({"model": model, "temperature": 0.2, "max_tokens": 3000,
                       "chat_template_kwargs": {"enable_thinking": False},
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    answer = live_translate.post("/v1/chat/completions", body, "application/json", timeout=600)
    if "error" in answer:
        raise RuntimeError(answer["error"].get("message", "the door refused the chat request"))
    return coerce_draft(extract_json(answer["choices"][0]["message"]["content"]))


def draft_notes(transcript: str, language: str, door: str, model: str, ask_fn=ask) -> dict:
    lang = LANGUAGES.get(language, "the language the meeting was held in")
    system = DRAFT_PROMPT.format(language=lang)
    parts = [ask_fn(door, model, system, c) for c in chunk(transcript, CHUNK_CHARS)]
    if len(parts) == 1:
        return parts[0]
    if not parts:
        return coerce_draft({})
    return ask_fn(door, model, COMBINE_PROMPT.format(language=lang),
                  "\n\n".join(json.dumps(p, ensure_ascii=False) for p in parts))


def asr_name() -> str:
    """The ASR meeting.sh will use, decided the way it decides; refuses whisper without a GPU."""
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0 \
        if shutil.which("nvidia-smi") else False
    asr = os.environ.get("ASR") or ("whisperx" if gpu else "tiiny")
    if asr == "whisperx" and not gpu:
        raise RuntimeError("ASR=whisperx but nvidia-smi sees no GPU: whisper never runs on a CPU here")
    return asr


def transcribe(audio: Path, language: str) -> str:
    sh = os.environ.get("MEETING_SH") or str(REPO / "scripts/meeting.sh")  # the override is for tests
    asr = asr_name()
    cmd = [sh, "--transcribe-only", str(audio)]
    if asr == "whisperx" and language in LANGUAGES:
        cmd += ["--language", language]
    run = subprocess.run(cmd, capture_output=True, text=True, env={**os.environ, "ASR": asr})
    if run.returncode:
        raise RuntimeError(f"meeting.sh failed ({run.returncode}): {run.stderr.strip()[-300:]}")
    lines = run.stdout.strip().splitlines()
    if not lines:
        raise RuntimeError("meeting.sh printed no transcript path")
    return read_transcript(Path(lines[-1]))


def write_results(put, folder: str, transcript: str, draft: dict) -> None:
    """The result files, `done` last so a reader never sees half of them."""
    put(folder + "transcript.txt", transcript.encode(), "text/plain")
    put(folder + "draft.json", json.dumps(draft, ensure_ascii=False, indent=2).encode(), "application/json")
    put(folder + "done", b"", "text/plain")


def process_folder(pod: Pod, folder: str, door: str, model: str) -> str:
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
        transcript = transcribe(audio, language)
    if not transcript.strip():
        raise RuntimeError("the transcript is empty")
    draft = draft_notes(transcript, language, door, model)
    write_results(pod.put, folder, transcript, draft)
    return "done"


def run_once(pod: Pod, door: str) -> int:
    """One pass over the inbox; the number of folders that failed."""
    failed = 0
    try:
        folders = [u for u in pod.children(pod.root + INBOX) if u.endswith("/")]
        model = meeting_inbox.running_box_model(door) if folders else ""
    except (urllib.error.URLError, OSError, KeyError, ValueError, SystemExit) as e:
        log(f"inbox not read: {type(e).__name__}: {e}")
        return 1
    for folder in folders:
        try:
            verdict = process_folder(pod, folder, door, model)
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
    assert coerce_draft(None) == coerce_draft("x") == {"summary": "", "next": [], "agreements": [], "lessons": []}
    d = coerce_draft({"summary": " ok ", "next": ["a", 3, None, "  ", " b "], "agreements": "no",
                      "lessons": ["x" * 999] * 40, "extra": 1})
    assert d["summary"] == "ok" and d["next"] == ["a", "b"] and d["agreements"] == []
    assert len(d["lessons"]) == ITEMS_MAX and len(d["lessons"][0]) == ITEM_MAX and set(d) == {
        "summary", "next", "agreements", "lessons"}
    assert len(coerce_draft({"summary": "s" * 9999})["summary"]) == SUMMARY_MAX
    assert extract_json('Here:\n```json\n{"summary": "x"}\n```') == {"summary": "x"}
    # A long transcript is asked in pieces, then combined once.
    calls = []
    fake = lambda door, model, system, user: (calls.append((system, user)), {"summary": "s", "next": [user[:1]]})[1]
    draft_notes("a\n" * CHUNK_CHARS, "fr", "d", "m", fake)
    assert len(calls) == 3 and "French" in calls[0][0] and calls[-1][0].startswith("Below are"), len(calls)
    calls.clear()
    draft_notes("short", "nl", "d", "m", fake)
    assert len(calls) == 1 and "Dutch" in calls[0][0]

    # The state decision per folder.
    meta = {"audio": "rec.webm"}
    assert decide(meta, {"meta.json", "rec.webm"}) == "process"
    assert decide(meta, {"meta.json", "rec.webm", "transcript.txt"}) == "process", "a half result is redone"
    assert decide(meta, {"meta.json", "rec.webm", "done"}).startswith("skip")
    assert decide(None, {"rec.webm"}).startswith("skip") and decide(meta, {"rec.webm"}).startswith("skip")
    assert decide(meta, {"meta.json"}).startswith("skip"), "audio still uploading"
    assert decide({"audio": 5}, {"meta.json"}).startswith("skip")

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
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else sys.exit(main())
