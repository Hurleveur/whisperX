#!/usr/bin/env python3
"""Turn the recordings dropped in a meetings folder into a transcript and a draft, unattended.

Run by the hai-os job runner with no arguments (see docs/meeting-inbox.md). It looks
for audio in `$MEETINGS_INBOX`, runs meeting.sh on each recording, writes the draft
and the transcript beside it in that folder, and moves the audio with the whisperX
files to `$HAI_RECORDINGS/<this host>/<recording name>/`, the archive. A recording
whose name has no date and time gets its modification time added first, so a file
name reused for every recording still names one meeting. With
`MEETINGS_INBOX` unset it does nothing: the folder is set on the one box that should
do the work, so two boxes sharing it never both take a recording.

    python scripts/meeting_inbox.py --selftest
"""

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOOR_FILE = Path.home() / ".config/hai/door"  # one line: the owner door's https URL
#: The tier hai-os ships this job on (background/jobs.toml): whichever model the box runs.
ANY_BOX = "ctx-any-local"
TRANSCRIPTS = REPO / "transcripts"  # where meeting.sh writes
# .mp4 and .webm: what WhatsApp and browsers save a voice recording as.
AUDIO = {".m4a", ".wav", ".mp3", ".mp4", ".webm", ".ogg", ".opus", ".flac", ".aac"}
# A file younger than this may still be recording or syncing in.
MIN_AGE_S = 120
#: The date and start time `note_name` reads out of a file name.
STAMP = re.compile(r"(\d{4})-(\d\d)-(\d\d)[ _T](\d\d)[_:.h-]?(\d\d)")


def recordings_root() -> Path:
    default = "~/Documents/GitHub/hai-os/.hai-shared/recordings"
    return Path(os.environ.get("HAI_RECORDINGS") or default).expanduser()


def inbox_dir() -> Path | None:
    value = os.environ.get("MEETINGS_INBOX")
    return Path(value).expanduser() if value else None


def transcripts_dir(inbox: Path) -> Path:
    """Where a recording's transcript goes: `MEETINGS_TRANSCRIPTS`, or beside the notes. A
    team folder synced to a shared pod would hand every word said to everyone in it."""
    value = os.environ.get("MEETINGS_TRANSCRIPTS")
    return Path(value).expanduser() if value else inbox


def pending(inbox: Path) -> list[Path]:
    """Audio in the inbox old enough to have finished recording or syncing in."""
    now = time.time()
    found = []
    for audio in sorted(inbox.iterdir()) if inbox.is_dir() else []:
        if audio.suffix.lower() not in AUDIO or not audio.is_file():
            continue
        if now - audio.stat().st_mtime < MIN_AGE_S:
            print(f"skip, still fresh: {audio.name}", file=sys.stderr)
            continue
        found.append(audio)
    return found


def running_box_model(door: str, listing: dict | None = None) -> str:
    """The chat model running on the owner's box, from the door's /v1/models."""
    if listing is None:
        with urllib.request.urlopen(f"{door}/v1/models", timeout=10) as resp:
            listing = json.load(resp)
    assert listing is not None
    running = [m["id"] for m in listing.get("data", []) if (m.get("hai_box") or {}).get("loaded")]
    if not running:
        raise SystemExit("no chat model runs on the owner's box (unplugged, locked or none loaded)")
    return running[0]


def llm_env() -> dict[str, str]:
    """The environment meeting.sh drafts with, which the merge uses too."""
    env = {**os.environ}
    # large-v2 is meeting.sh's default; turbo is the speed/accuracy trade for a job
    # that owns the GPU for the duration.
    env.setdefault("WHISPER_MODEL", "large-v3-turbo")
    # The Background panel's pick (the runner's HAI_JOB_MODEL): a model on the owner's
    # box, named (`box/<box>/<model>`) or whichever runs there (`ctx-any-local`), asked
    # through the owner door, which answers any of his boxes on the tailnet by its
    # Tailscale identity: no bearer token goes there, the transcript never leaves his
    # machines, and unlike hai-context's own `any-local` it never falls back to this
    # box's CPU model, so a box that is off fails the run and the next one retries.
    picked = env.get("HAI_JOB_MODEL", "")
    if picked.startswith("box/") or picked == ANY_BOX:
        door = DOOR_FILE.read_text().strip().rstrip("/")
        env.setdefault("LLM_API", "openai")
        env.setdefault("LLM_HOST", f"{door}/v1")
        env.setdefault("LLM_MODEL", picked if picked != ANY_BOX else running_box_model(door))
        env.setdefault("LLM_TOKEN_FILE", "")
        env.setdefault("LLM_ESCALATE", "")
    elif picked:
        raise SystemExit(f"meeting-inbox drafts on a box model ({ANY_BOX} or box/...), not {picked!r}")
    # hai-context's local model, the only one a background job may use. The bearer
    # token is read from this file by analyze_meeting.py, never passed on a command line.
    env.setdefault("LLM_API", "openai")
    env.setdefault("LLM_HOST", "http://127.0.0.1:18701/v1")
    env.setdefault("LLM_MODEL", "hai-local")
    env.setdefault("LLM_TOKEN_FILE", str(Path("~/.config/hai/tokens/scheduler").expanduser()))
    # Drafted on hai-context's `meeting` cloud purpose (a ZDR Gemma, metered under the
    # owner's caps): the local 4B model invented the claims beside real quotes. Set
    # LLM_ESCALATE to an empty string to keep the transcript on the box.
    env.setdefault("LLM_ESCALATE", "meeting")
    return env


def run_meeting(audio: Path, transcribe_only: bool = False) -> None:
    """meeting.sh does the work and resumes from transcripts/ after a crash."""
    env = llm_env()
    flags = ["--transcribe-only"] if transcribe_only else []
    subprocess.run([str(REPO / "scripts" / "meeting.sh"), *flags, str(audio)], env=env, check=True)


def note_name(audio: Path) -> str:
    """`10-21-26 10h30`: the vault's M-DD-YY meeting names, plus the start time so two
    recordings of one day stay apart. The date in the file name wins; the mtime is
    when the recording stopped, not when it began."""
    m = STAMP.search(audio.stem)
    if m:
        y, mo, d, hh, mm = m.groups()
    else:
        y, mo, d, hh, mm = time.strftime("%Y %m %d %H %M", time.localtime(audio.stat().st_mtime)).split()
    return f"{int(mo)}-{d}-{y[2:]} {hh}h{mm}"


def model_args() -> argparse.Namespace:
    env = llm_env()
    return argparse.Namespace(
        api=env["LLM_API"], host=env["LLM_HOST"], model=env["LLM_MODEL"], ctx=32768,
        token_file=Path(env["LLM_TOKEN_FILE"]) if env.get("LLM_TOKEN_FILE") else None,
        escalate=env.get("LLM_ESCALATE", ""),
    )


#: A photo's reading is a page at most; the cap is what ends a loop.
PHOTO_TOKENS = 1500


def once_each(text: str) -> str:
    """A line the model repeated straight after itself kept once: reading a whiteboard,
    the Tiiny's Qwen3.6-35B wrote one word on a line ~1000 times (2026-11-09)."""
    out = []
    for line in text.splitlines():
        if not (out and line.strip() and line.strip() == out[-1].strip()):
            out.append(line)
    return "\n".join(out)


def read_photos(photos: list[Path]) -> dict[str, str]:
    """What each photo shows, by file name, one request per photo: two images in one
    request stopped the Tiiny's Qwen3.6-35B twice (2026-11-09) where one alone answered.
    A photo the model fails on is left out with a line on stderr."""
    import analyze_meeting

    args = model_args()
    if args.escalate:  # the cloud lane rebuilds the messages as plain text
        return {}
    seen = {}
    for photo in photos:
        try:
            seen[photo.name] = once_each(analyze_meeting.generate(
                "This photo was taken during a team meeting (a whiteboard, a slide, a sketch "
                "or a page). Transcribe every word written on it exactly, then say in a few "
                "lines what it shows and how its parts connect. Nothing else.",
                args, [photo], limit=PHOTO_TOKENS))
        except (OSError, ValueError, KeyError) as exc:
            print(f"photo {photo.name} not read, left out: {exc}", file=sys.stderr)
    return seen


def merge_note(note: str, draft: str, photos: dict[str, str] | None = None) -> str:
    """The notes that go under the team's note: the recording's draft worked together with
    what the note and its photos (`read_photos`) say. The note itself is put back above by
    the caller."""
    import analyze_meeting  # beside this script; only a merge needs it

    # Said agreements are verbatim lines kept for checking ("Agreed?"), noise in the team's
    # note; they stay in the archived summary.md.
    draft = draft.split("\n## Said agreements", 1)[0].strip()
    shown = "".join(f"--- PHOTO {name} ---\n{text}\n--- END PHOTO ---\n\n"
                    for name, text in (photos or {}).items())
    answer = analyze_meeting.generate(
        "You write a team's meeting notes. Below is the note the team wrote by hand, "
        + ("what the photos it embeds show, " if shown else "")
        + "then notes drafted by a machine from a recording of that day's meeting.\n\n"
        f"--- TEAM NOTE ---\n{note}\n--- END TEAM NOTE ---\n\n"
        f"{shown}"
        f"--- DRAFT FROM THE RECORDING ---\n{draft}\n--- END DRAFT ---\n\n"
        "Rules. The team note stays above your answer exactly as it is: do not copy it. "
        "Write the meeting notes that go under it, under the draft's own headings, starting "
        "with `## Summary`: what the team will still need in a month, worked together "
        "with what the team note and the photos add. Shorter than the draft, never longer. Cut what the draft kept that has no "
        "use later: the day's logistics, small talk, anecdotes, demo narration. "
        + ("Add a section `## From the photos`: for each photo, two or three sentences on "
           "the ideas it holds and how they tie to what was said; never list every word "
           "on it. " if shown else "")
        + "Where the team note names the people, write a name in place of a SPEAKER label "
        "only when the note and the draft make clear who it is. Keep the draft's quotes "
        "exactly. Write in the team note's language. Output the notes only.",
        model_args(),
    )
    return without_note(note, strip_markers(answer))


#: The least of the draft's words a merge may keep. Told to write less than the draft, the
#: Tiiny's Qwen3.6-35B once replaced a 480-word draft with the team note's one line (2026-11-09).
KEPT_SHARE = 0.3


def kept_enough(notes: str, draft: str) -> bool:
    """Whether the merged notes are long enough to still hold the draft's content."""
    draft = draft.split("\n## Said agreements", 1)[0]
    return len(notes.split()) >= KEPT_SHARE * len(draft.split())


def without_note(note: str, answer: str) -> str:
    """The answer minus a copy of the note it opens with: told not to, the Tiiny's
    Qwen3.6-35B still began with the whole note (2026-11-09), which the caller puts above."""
    want = [line.strip() for line in note.splitlines() if line.strip()]
    lines = answer.splitlines()
    i = 0
    for line in lines:
        if i == len(want) or (line.strip() and line.strip() != want[i]):
            break
        i += bool(line.strip())
    if i < len(want):
        return answer
    rest, seen = [], 0
    for line in lines:
        if seen < len(want):
            seen += bool(line.strip())
            continue
        rest.append(line)
    return "\n".join(rest).strip()


#: An image the note embeds, Obsidian's `![[name]]` or Markdown's `![](path)`.
EMBED = re.compile(r"!\[\[([^\]|#]+)[^\]]*\]\]|!\[[^\]]*\]\(<?([^)>]+?)>?\)")
PICTURE = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
MAX_PHOTOS = 8


def photos(note: str, where: Path) -> list[Path]:
    """The images the note at `where` embeds, found the way Obsidian finds them: beside
    the note, then by name anywhere in the vault (the folder holding `.obsidian`)."""
    vault = next((d for d in where.parents if (d / ".obsidian").is_dir()), where.parent)
    found = []
    for m in EMBED.finditer(note):
        name = (m.group(1) or m.group(2)).strip()
        if Path(name).suffix.lower() not in PICTURE:
            continue
        hit = where.parent / name
        if not hit.is_file():
            hit = next((p for p in vault.rglob(Path(name).name) if p.is_file()), None)
        if hit is None:
            print(f"photo not found, left out: {name}", file=sys.stderr)
        elif hit not in found:
            found.append(hit)
    return found[:MAX_PHOTOS]


#: The prompt's own fences, which the model echoes back (seen live, 2026-10-28).
MARKERS = re.compile(r"^--- (END )?(TEAM NOTE|DRAFT FROM THE RECORDING|PHOTO.*?) ---[ \t]*\n?", re.MULTILINE)


def strip_markers(answer: str) -> str:
    return MARKERS.sub("", answer)


def stamped(audio: Path, archive: Path) -> Path:
    """The recording renamed with its time when its name carries none, so a recorder
    that saves `Recording.m4a` every time still gives each meeting its own transcript
    and archive folder: both are keyed by the file name. Never onto a name in use in
    the folder or the archive."""
    if STAMP.search(audio.stem):
        return audio
    base = f"{audio.stem} {time.strftime('%Y-%m-%d %H_%M', time.localtime(audio.stat().st_mtime))}"
    target, n = audio.with_name(base + audio.suffix), 2
    while target.exists() or (archive / target.stem).exists():
        target, n = audio.with_name(f"{base} {n}{audio.suffix}"), n + 1
    return audio.rename(target)


def glob_escape(s: str) -> str:
    return "".join(f"[{c}]" if c in "*?[" else c for c in s)


def merge_into(team: Path, draft: str, out: Path) -> bool:
    """Put the recording's notes under the team's note for that day, or leave it alone.

    What the team wrote stays first, byte for byte: the model only writes what goes
    under it. Nothing is written if the note changed while the model ran; the note as
    it was goes to `out` first."""
    before, seen = team.read_text(encoding="utf-8"), team.stat().st_mtime_ns
    try:
        read = read_photos(photos(before, team))
        if read:  # kept as it came: the notes can be redone from it without the model
            (out / "photos.md").write_text("".join(f"## {n}\n\n{t}\n\n" for n, t in read.items()),
                                           encoding="utf-8")
        notes = merge_note(before, draft, read)
    except (OSError, ValueError, KeyError) as exc:
        print(f"merge into {team.name} failed: {exc}", file=sys.stderr)
        return False
    if not notes.strip():
        print(f"merge into {team.name} refused: the model wrote nothing", file=sys.stderr)
        return False
    if not kept_enough(notes, draft):
        print(f"merge into {team.name} refused: the notes lost most of the draft", file=sys.stderr)
        return False
    if team.stat().st_mtime_ns != seen:
        print(f"merge into {team.name} refused: the note changed while merging", file=sys.stderr)
        return False
    shutil.copy2(team, out / f"{team.stem} before merge.md")
    tmp = team.with_name(f".{team.name}.merge")
    tmp.write_text(before.rstrip() + "\n\n" + notes.strip() + "\n", encoding="utf-8")
    tmp.replace(team)
    return True


def publish(audio: Path, out: Path) -> str:
    """Notes beside the audio, the pipeline's files into `out`, then the audio itself:
    `summary.md` in `out` marks the notes written, the move marks the recording done.
    A team note already written for that day (`M-DD-YY.md`) gets the draft merged in
    instead of a draft beside it. Returns the line for the runner's report."""
    stem = audio.stem
    draft = TRANSCRIPTS / f"{stem}-notes" / "summary.md"
    if not draft.is_file() or not (TRANSCRIPTS / f"{stem}.json").is_file():
        raise FileNotFoundError(f"meeting.sh left no transcript or summary for {stem}")
    out.mkdir(parents=True, exist_ok=True)
    for f in TRANSCRIPTS.glob(f"{glob_escape(stem)}.*"):
        shutil.copy2(f, out / f.name)
    # Never over an existing note: two recordings of one minute, or a draft already edited.
    base, n, name = note_name(audio), 2, note_name(audio)
    said = transcripts_dir(audio.parent)
    while (audio.parent / f"{name} draft.md").exists() or (said / f"{name} transcript.md").exists():
        name, n = f"{base} {n}", n + 1
    team = audio.parent / f"{base.split()[0]}.md"
    if team.is_file() and merge_into(team, draft.read_text(encoding="utf-8"), out):
        line = f"Meeting notes merged into {team.stem}"
    else:
        shutil.copy2(draft, audio.parent / f"{name} draft.md")
        line = f"Meeting notes drafted: {name}"
    said.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TRANSCRIPTS / f"{stem}.txt", said / f"{name} transcript.md")
    tmp = out / "summary.md.tmp"
    shutil.copy2(draft, tmp)
    tmp.replace(out / "summary.md")
    archive(audio, out)
    return line


def archive(audio: Path, out: Path) -> None:
    """Move the recording out of the shared folder; a name already archived keeps both."""
    target, n = out / audio.name, 2
    while target.exists():
        target, n = out / f"{audio.stem} {n}{audio.suffix}", n + 1
    shutil.move(audio, target)


def report(line: str) -> None:
    """One line for the hai-os runner's quiet "finished" notice (`HAI_JOB_REPORT`);
    nothing when run by hand."""
    path = os.environ.get("HAI_JOB_REPORT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> int:
    inbox = inbox_dir()
    if inbox is None:
        print("MEETINGS_INBOX is not set on this box: nothing to do", file=sys.stderr)
        return 0
    if not inbox.is_dir():
        print(f"FAILED: the meetings folder {inbox} does not exist", file=sys.stderr)
        return 1
    root, me = recordings_root(), os.environ.get("HAI_HOSTNAME") or socket.gethostname()
    todo, failed = [], 0
    for audio in pending(inbox):
        try:
            audio = stamped(audio, root / me)
        except OSError as exc:
            failed += 1
            print(f"FAILED to rename {audio.name}: {exc}", file=sys.stderr)
            continue
        out = root / me / audio.stem
        if not (out / "summary.md").exists():
            todo.append(audio)
            continue
        # Notes written, move failed last time: finish it, never redraft over the notes.
        try:
            archive(audio, out)
        except OSError as exc:
            failed += 1
            print(f"FAILED to archive {audio.name}: {exc}", file=sys.stderr)
    print(f"{len(todo)} recording(s) to do in {inbox}", file=sys.stderr)
    # Every transcript before any draft: the local model holds ~3.4 GB of a 6 GB card
    # for minutes after a draft, and the next recording's diarization ran out of
    # memory beside it (2026-10-21).
    for audio in list(todo):
        try:
            run_meeting(audio, transcribe_only=True)
        except (subprocess.CalledProcessError, OSError) as exc:
            failed += 1
            todo.remove(audio)
            print(f"FAILED {audio.name}: {exc}", file=sys.stderr)
    for audio in todo:
        try:
            run_meeting(audio)
            line = publish(audio, root / me / audio.stem)
            print(f"done: {audio.name}", file=sys.stderr)
            report(line)
        except (subprocess.CalledProcessError, OSError) as exc:
            failed += 1
            print(f"FAILED {audio.name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


def selftest() -> None:
    global TRANSCRIPTS
    tmp = Path(tempfile.mkdtemp())
    inbox, root, TRANSCRIPTS = tmp / "meetings", tmp / "rec", tmp / "tr"
    inbox.mkdir()
    global DOOR_FILE
    DOOR_FILE = tmp / "door"
    DOOR_FILE.write_text("https://door.example:1/\n")
    os.environ["HAI_JOB_MODEL"] = "box/tiiny/Qwen/Qwen3.6-35B-A3B"
    env = llm_env()
    assert (env["LLM_HOST"], env["LLM_MODEL"]) == ("https://door.example:1/v1", "box/tiiny/Qwen/Qwen3.6-35B-A3B")
    assert env["LLM_TOKEN_FILE"] == env["LLM_ESCALATE"] == "", "no token to the door, no cloud"
    os.environ["HAI_JOB_MODEL"] = "sonnet"
    try:
        llm_env()
        raise AssertionError("a pick it cannot use is refused, not ignored")
    except SystemExit:
        pass
    del os.environ["HAI_JOB_MODEL"]
    assert llm_env()["LLM_MODEL"] == "hai-local"
    listing = {"data": [{"id": "local", "hai_own": {}},
                        {"id": "box/t/a", "hai_box": {"loaded": False}},
                        {"id": "box/t/b", "hai_box": {"loaded": True}}]}
    assert running_box_model("", listing) == "box/t/b"
    try:
        running_box_model("", {"data": listing["data"][:2]})
        raise AssertionError("nothing running on the box: the run fails, never this box's own model")
    except SystemExit:
        pass
    for name in ["m [1].m4a", "notes.md", "bad.wav", "voice.mp4"]:
        f = inbox / name
        f.write_bytes(b"x")
        os.utime(f, (0, 0))
    (inbox / "fresh.mp3").write_bytes(b"x")  # still syncing in: left for the next run
    assert [p.name for p in pending(inbox)] == ["bad.wav", "m [1].m4a", "voice.mp4"]
    calls = []

    def fake(audio: Path, transcribe_only: bool = False) -> None:
        calls.append(audio.name + ("/t" if transcribe_only else ""))
        if audio.stem.startswith("bad "):
            raise subprocess.CalledProcessError(1, "meeting.sh")
        if transcribe_only:
            return
        (TRANSCRIPTS / f"{audio.stem}-notes").mkdir(parents=True, exist_ok=True)
        (TRANSCRIPTS / f"{audio.stem}.json").write_text("{}")
        (TRANSCRIPTS / f"{audio.stem}.txt").write_text("hello")
        (TRANSCRIPTS / f"{audio.stem}-notes" / "summary.md").write_text("draft")

    globals()["run_meeting"] = fake
    os.environ.pop("MEETINGS_INBOX", None)
    assert main() == 0 and calls == []  # no folder set on this box: nothing happens
    os.environ["MEETINGS_INBOX"] = str(tmp / "missing")
    assert main() == 1 and calls == []
    os.environ["MEETINGS_INBOX"] = str(inbox)
    os.environ["HAI_RECORDINGS"], os.environ["HAI_HOSTNAME"] = str(root), "me"
    os.environ["HAI_JOB_REPORT"] = str(tmp / "report.txt")
    epoch = note_name(inbox / "m [1].m4a")  # no date in the name: from the mtime, pinned to the epoch
    st = time.strftime("%Y-%m-%d %H_%M", time.localtime(0))  # what a name without one is stamped with
    assert main() == 1  # bad.wav failed, the others were still done
    assert (tmp / "report.txt").read_text() == f"Meeting notes drafted: {epoch}\nMeeting notes drafted: {epoch} 2\n"
    # Every transcription runs before any draft, and a failed one is not drafted.
    assert calls == [f"bad {st}.wav/t", f"m [1] {st}.m4a/t", f"voice {st}.mp4/t",
                     f"m [1] {st}.m4a", f"voice {st}.mp4"], calls
    out = root / "me" / f"m [1] {st}"
    assert (out / "summary.md").read_text() == "draft" and (out / f"m [1] {st}.txt").exists()
    # The notes land beside the audio; the audio leaves the folder for the archive.
    assert (inbox / f"{epoch} draft.md").read_text() == "draft"
    assert (inbox / f"{epoch} transcript.md").read_text() == "hello"
    assert (inbox / f"{epoch} 2 draft.md").exists()  # same minute: numbered, not written over
    assert not (inbox / "m [1].m4a").exists() and (out / f"m [1] {st}.m4a").read_bytes() == b"x"
    assert sorted(p.name for p in inbox.iterdir() if p.suffix != ".md") == [f"bad {st}.wav", "fresh.mp3"]
    assert (inbox / "notes.md").exists() and not (root / "me" / f"bad {st}" / "summary.md").exists()
    assert note_name(Path("Sound record (2026-10-21 10_30_15).m4a")) == "10-21-26 10h30"
    assert note_name(Path("rec 2026-09-02T08:05.wav")) == "9-02-26 08h05"
    calls.clear()
    assert main() == 1 and calls == [f"bad {st}.wav/t"]  # archived ones are gone, the failed one retried
    # A recording whose notes exist but whose move failed is moved, never redrafted.
    (inbox / f"voice {st}.mp4").write_bytes(b"y")
    os.utime(inbox / f"voice {st}.mp4", (0, 0))
    (inbox / f"{epoch} draft.md").write_text("edited by hand")
    calls.clear()
    assert main() == 1 and calls == [f"bad {st}.wav/t"]
    assert (inbox / f"{epoch} draft.md").read_text() == "edited by hand"
    assert (root / "me" / f"voice {st}" / f"voice {st} 2.mp4").read_bytes() == b"y"
    # A recorder that reuses its file name: a new recording, never taken for the old one.
    (inbox / "voice.mp4").write_bytes(b"z")
    os.utime(inbox / "voice.mp4", (0, 0))
    calls.clear()
    assert main() == 1 and calls == [f"bad {st}.wav/t", f"voice {st} 2.mp4/t", f"voice {st} 2.mp4"], calls
    assert (root / "me" / f"voice {st} 2" / f"voice {st} 2.mp4").read_bytes() == b"z"
    assert (tmp / "report.txt").read_text().endswith(f"Meeting notes drafted: {epoch} 3\n")

    # A team note for that day keeps what the team wrote first, untouched, and gets the
    # recording's notes under it; the note as it was is archived.
    assert strip_markers("--- TEAM NOTE ---\n# T\n- a\n--- END TEAM NOTE ---\n") == "# T\n- a\n"
    team, answers, shown = inbox / "10-21-26.md", [], []
    team.write_text("---\nparticipants: A, B\n---\n# Team\n- said by hand\n![[board.jpg]]\n")
    (tmp / ".obsidian").mkdir()
    (tmp / "Attachments").mkdir()
    (tmp / "Attachments" / "board.jpg").write_bytes(b"jpg")

    def fake_merge(note: str, draft: str, photos: dict[str, str] | None = None) -> str:
        shown.append(photos)
        return answers.pop(0)

    globals()["merge_note"] = fake_merge
    globals()["read_photos"] = lambda found: {p.name: f"what {p.name} shows" for p in found}
    for stem, answer in [("good", "- from the recording\n- what the board shows\n"), ("nothing", "  \n")]:
        audio = inbox / f"{stem} 2026-10-21 10_30.m4a"
        audio.write_bytes(b"x")
        os.utime(audio, (0, 0))
        answers.append(answer)
    (tmp / "report.txt").write_text("")
    calls.clear()
    os.environ["MEETINGS_TRANSCRIPTS"] = str(tmp / "mine")
    assert main() == 1  # bad.wav still failing
    os.environ.pop("MEETINGS_TRANSCRIPTS")
    assert team.read_text() == (
        "---\nparticipants: A, B\n---\n# Team\n- said by hand\n![[board.jpg]]\n\n"
        "- from the recording\n- what the board shows\n"
    ), team.read_text()
    assert shown[0] == {"board.jpg": "what board.jpg shows"}  # found in the vault, read
    assert (root / "me" / "good 2026-10-21 10_30" / "photos.md").read_text() == (
        "## board.jpg\n\nwhat board.jpg shows\n\n")
    assert strip_markers("--- PHOTO a.jpg ---\nx\n--- END PHOTO ---\n") == "x\n"
    assert without_note("---\nA\n---\nhi", "---\nA\n---\n\nhi\n\n## Summary\nx") == "## Summary\nx"
    assert without_note("---\nA\n---\nhi", "## Summary\n---\nA") == "## Summary\n---\nA"  # no copy
    assert without_note("hi\nthere", "hi\n## Summary") == "hi\n## Summary"  # a partial copy is kept
    assert kept_enough("a b c", "w " * 10) and not kept_enough("a b", "w " * 10)
    assert kept_enough("a b c", "w " * 10 + "\n## Said agreements\n" + "x " * 90)
    assert once_each("Hyper\nDyscic\nDyscic\n  Dyscic\n\n\nend") == "Hyper\nDyscic\n\n\nend"
    assert (root / "me" / "good 2026-10-21 10_30" / "10-21-26 before merge.md").read_text().endswith("![[board.jpg]]\n")
    # An empty answer is refused: the note stands and the draft lands beside it.
    assert (inbox / "10-21-26 10h30 2 draft.md").read_text() == "draft"
    assert not (inbox / "10-21-26 10h30 draft.md").exists()  # the merged one wrote no draft
    assert (tmp / "mine" / "10-21-26 10h30 transcript.md").exists()  # MEETINGS_TRANSCRIPTS
    assert not (inbox / "10-21-26 10h30 transcript.md").exists()
    assert (tmp / "report.txt").read_text() == (
        "Meeting notes merged into 10-21-26\nMeeting notes drafted: 10-21-26 10h30 2\n"
    ), (tmp / "report.txt").read_text()
    # Embeds of both kinds, beside the note or anywhere in the vault; others left out.
    (inbox / "beside.png").write_bytes(b"png")
    note = "![[board.jpg|300]] ![](beside.png) ![[talk.pdf]] ![[gone.jpg]] ![[board.jpg]]"
    assert [p.name for p in photos(note, team)] == ["board.jpg", "beside.png"]
    shutil.rmtree(tmp)
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else sys.exit(main())
