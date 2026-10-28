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
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
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


def llm_env() -> dict[str, str]:
    """The environment meeting.sh drafts with, which the merge uses too."""
    env = {**os.environ}
    # large-v2 is meeting.sh's default; turbo is the speed/accuracy trade for a job
    # that owns the GPU for the duration.
    env.setdefault("WHISPER_MODEL", "large-v3-turbo")
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


def merge_note(note: str, draft: str) -> str:
    """The team's note for that day with the recording's draft worked in, by the model."""
    import analyze_meeting  # beside this script; only a merge needs it

    env = llm_env()
    args = argparse.Namespace(
        api=env["LLM_API"], host=env["LLM_HOST"], model=env["LLM_MODEL"], ctx=32768,
        token_file=Path(env["LLM_TOKEN_FILE"]) if env.get("LLM_TOKEN_FILE") else None,
        escalate=env.get("LLM_ESCALATE", ""),
    )
    # The note before the draft: the model reads what the team already wrote, its
    # headings and its people, before it sees what to add.
    answer = analyze_meeting.generate(
        "You keep a team's meeting note. Below is the note the team wrote, then notes "
        "drafted by a machine from a recording of that day's meeting.\n\n"
        f"--- TEAM NOTE ---\n{note}\n--- END TEAM NOTE ---\n\n"
        f"--- DRAFT FROM THE RECORDING ---\n{draft}\n--- END DRAFT ---\n\n"
        "Rules. Read the team note first: its headings, its people, what it already says. "
        "Output the whole team note with what the draft adds worked in where it belongs. "
        "Copy every line of the team note exactly, in its order. Never reword, move or "
        "delete one. Add only what the draft says and the note does not. Already in the "
        "note - skip. Keep the draft's quotes exactly. Write additions in the note's "
        "language. Output the note only.",
        args,
    )
    return strip_markers(answer)


#: The prompt's own fences, which the model echoes back (seen live, 2026-10-28).
MARKERS = re.compile(r"^--- (END )?(TEAM NOTE|DRAFT FROM THE RECORDING) ---[ \t]*\n?", re.MULTILINE)


def strip_markers(answer: str) -> str:
    return MARKERS.sub("", answer)


def kept_whole(note: str, merged: str) -> bool:
    """Every non-empty line of the note is still in the merge, in the same order."""
    rest = iter(line.strip() for line in merged.splitlines())
    return all(any(line == m for m in rest) for line in (l.strip() for l in note.splitlines()) if line)


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
    """Work the draft into the team's note for that day, or leave the note alone.

    The merge is written only when it keeps every line the team wrote and nobody
    edited the note while the model ran; the note as it was goes to `out` first."""
    before, seen = team.read_text(encoding="utf-8"), team.stat().st_mtime_ns
    try:
        merged = merge_note(before, draft)
    except (OSError, ValueError, KeyError) as exc:
        print(f"merge into {team.name} failed: {exc}", file=sys.stderr)
        return False
    if not merged.strip() or not kept_whole(before, merged):
        print(f"merge into {team.name} refused: it dropped or changed a line of the note", file=sys.stderr)
        return False
    if team.stat().st_mtime_ns != seen:
        print(f"merge into {team.name} refused: the note changed while merging", file=sys.stderr)
        return False
    shutil.copy2(team, out / f"{team.stem} before merge.md")
    tmp = team.with_name(f".{team.name}.merge")
    tmp.write_text(merged.rstrip() + "\n", encoding="utf-8")
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
    while (audio.parent / f"{name} draft.md").exists() or (audio.parent / f"{name} transcript.md").exists():
        name, n = f"{base} {n}", n + 1
    team = audio.parent / f"{base.split()[0]}.md"
    if team.is_file() and merge_into(team, draft.read_text(encoding="utf-8"), out):
        line = f"Meeting notes merged into {team.stem}"
    else:
        shutil.copy2(draft, audio.parent / f"{name} draft.md")
        line = f"Meeting notes drafted: {name}"
    shutil.copy2(TRANSCRIPTS / f"{stem}.txt", audio.parent / f"{name} transcript.md")
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

    # A team note for that day gets the draft merged in, the note as it was archived.
    assert kept_whole("# T\n\n- a\n- b\n", "# T\n- a\n- new\n  - b  \n")
    assert not kept_whole("# T\n- a\n- b\n", "# T\n- b\n- a\n")  # reordered
    assert not kept_whole("# T\n- a\n- b\n", "# T\n- a\n")  # dropped
    assert strip_markers("--- TEAM NOTE ---\n# T\n- a\n--- END TEAM NOTE ---\n") == "# T\n- a\n"
    team, answers = inbox / "10-21-26.md", []
    team.write_text("# Team\n- said by hand\n")
    globals()["merge_note"] = lambda note, draft: answers.pop(0)
    for stem, answer in [("good", "# Team\n- said by hand\n- from the recording\n"),
                         ("lossy", "# Team\n- from the recording\n")]:
        audio = inbox / f"{stem} 2026-10-21 10_30.m4a"
        audio.write_bytes(b"x")
        os.utime(audio, (0, 0))
        answers.append(answer)
    (tmp / "report.txt").write_text("")
    calls.clear()
    assert main() == 1  # bad.wav still failing
    assert team.read_text() == "# Team\n- said by hand\n- from the recording\n"
    assert (root / "me" / "good 2026-10-21 10_30" / "10-21-26 before merge.md").read_text() == "# Team\n- said by hand\n"
    # The lossy merge was refused: the note stands and the draft lands beside it.
    assert (inbox / "10-21-26 10h30 2 draft.md").read_text() == "draft"
    assert not (inbox / "10-21-26 10h30 draft.md").exists()  # the merged one wrote no draft
    assert (inbox / "10-21-26 10h30 transcript.md").exists()
    assert (tmp / "report.txt").read_text() == (
        "Meeting notes merged into 10-21-26\nMeeting notes drafted: 10-21-26 10h30 2\n"
    ), (tmp / "report.txt").read_text()
    shutil.rmtree(tmp)
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else sys.exit(main())
