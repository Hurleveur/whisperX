#!/usr/bin/env python3
"""Turn other boxes' meeting recordings into a transcript and a draft, unattended.

Run by the hai-os job runner with no arguments (see docs/meeting-inbox.md). It looks
in `$HAI_RECORDINGS/<host>/` for every host except this one, runs meeting.sh on each
recording that has no `summary.md` yet, and writes the results to
`$HAI_RECORDINGS/<this host>/<recording name>/`. A box only ever writes its own
folder, so Syncthing never has two writers; the inbox files are never touched.

    python scripts/meeting_inbox.py --selftest
"""

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
AUDIO = {".m4a", ".wav", ".mp3", ".ogg", ".opus", ".flac", ".aac"}
# A file younger than this may still be recording or syncing in.
MIN_AGE_S = 120


def recordings_root() -> Path:
    default = "~/Documents/GitHub/hai-os/.hai-shared/recordings"
    return Path(os.environ.get("HAI_RECORDINGS") or default).expanduser()


def vault_dir() -> Path:
    default = "~/Documents/GitHub/sandbox-office-vault/team/alexandre/meetings"
    return Path(os.environ.get("MEETINGS_VAULT_DIR") or default).expanduser()


def pending(root: Path, me: str) -> list[Path]:
    """Audio in every other host's folder whose output has no summary.md yet."""
    now = time.time()
    found = []
    hosts = sorted(p for p in root.iterdir() if p.is_dir() and p.name != me) if root.is_dir() else []
    for host in hosts:
        for audio in sorted(host.iterdir()):
            if audio.suffix.lower() not in AUDIO or not audio.is_file():
                continue
            if (root / me / audio.stem / "summary.md").exists():
                continue
            if now - audio.stat().st_mtime < MIN_AGE_S:
                print(f"skip, still fresh: {audio.name}", file=sys.stderr)
                continue
            found.append(audio)
    return found


def run_meeting(audio: Path, transcribe_only: bool = False) -> None:
    """meeting.sh does the work and resumes from transcripts/ after a crash."""
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
    flags = ["--transcribe-only"] if transcribe_only else []
    subprocess.run([str(REPO / "scripts" / "meeting.sh"), *flags, str(audio)], env=env, check=True)


def note_name(audio: Path) -> str:
    """`10-21-26 10h30`: the vault's M-DD-YY meeting names, plus the start time so two
    recordings of one day stay apart. The date in the file name wins; the mtime is
    when the recording stopped, not when it began."""
    m = re.search(r"(\d{4})-(\d\d)-(\d\d)[ _T](\d\d)[_:.h-]?(\d\d)", audio.stem)
    if m:
        y, mo, d, hh, mm = m.groups()
    else:
        y, mo, d, hh, mm = time.strftime("%Y %m %d %H %M", time.localtime(audio.stat().st_mtime)).split()
    return f"{int(mo)}-{d}-{y[2:]} {hh}h{mm}"


def glob_escape(s: str) -> str:
    return "".join(f"[{c}]" if c in "*?[" else c for c in s)


def publish(audio: Path, out: Path) -> None:
    """Copy the pipeline's files into `out`; summary.md goes last and marks it done."""
    stem = audio.stem
    draft = TRANSCRIPTS / f"{stem}-notes" / "summary.md"
    if not draft.is_file() or not (TRANSCRIPTS / f"{stem}.json").is_file():
        raise FileNotFoundError(f"meeting.sh left no transcript or summary for {stem}")
    out.mkdir(parents=True, exist_ok=True)
    for f in TRANSCRIPTS.glob(f"{glob_escape(stem)}.*"):
        shutil.copy2(f, out / f.name)
    # Vault first: once summary.md exists the recording is never visited again.
    vault = vault_dir()
    if vault.is_dir():
        name = note_name(audio)
        shutil.copy2(draft, vault / f"{name} draft.md")
        shutil.copy2(TRANSCRIPTS / f"{stem}.txt", vault / f"{name} transcript.md")
    else:
        print(f"no vault folder {vault}, skipping the copy", file=sys.stderr)
    tmp = out / "summary.md.tmp"
    shutil.copy2(draft, tmp)
    tmp.replace(out / "summary.md")


def report(line: str) -> None:
    """One line for the hai-os runner's quiet "finished" notice (`HAI_JOB_REPORT`);
    nothing when run by hand."""
    path = os.environ.get("HAI_JOB_REPORT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def main() -> int:
    root, me = recordings_root(), os.environ.get("HAI_HOSTNAME") or socket.gethostname()
    todo = pending(root, me)
    print(f"{len(todo)} recording(s) to do for {me}", file=sys.stderr)
    failed = 0
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
            publish(audio, root / me / audio.stem)
            print(f"done: {audio.name}", file=sys.stderr)
            report(f"Meeting notes drafted: {note_name(audio)}")
        except (subprocess.CalledProcessError, OSError) as exc:
            failed += 1
            print(f"FAILED {audio.name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


def selftest() -> None:
    global TRANSCRIPTS
    tmp = Path(tempfile.mkdtemp())
    root, TRANSCRIPTS = tmp / "rec", tmp / "tr"
    os.environ["MEETINGS_VAULT_DIR"] = str(tmp / "vault")
    (tmp / "vault").mkdir()
    for host, name in [("a", "m [1].m4a"), ("a", "notes.txt"), ("me", "own.m4a"), ("a", "bad.wav")]:
        (root / host).mkdir(parents=True, exist_ok=True)
        f = root / host / name
        f.write_bytes(b"x")
        os.utime(f, (0, 0))
    assert [p.name for p in pending(root, "me")] == ["bad.wav", "m [1].m4a"]
    calls = []

    def fake(audio: Path, transcribe_only: bool = False) -> None:
        calls.append(audio.name + ("/t" if transcribe_only else ""))
        if audio.stem == "bad":
            raise subprocess.CalledProcessError(1, "meeting.sh")
        if transcribe_only:
            return
        (TRANSCRIPTS / f"{audio.stem}-notes").mkdir(parents=True, exist_ok=True)
        (TRANSCRIPTS / f"{audio.stem}.json").write_text("{}")
        (TRANSCRIPTS / f"{audio.stem}.txt").write_text("hello")
        (TRANSCRIPTS / f"{audio.stem}-notes" / "summary.md").write_text("draft")

    globals()["run_meeting"] = fake
    os.environ["HAI_RECORDINGS"], os.environ["HAI_HOSTNAME"] = str(root), "me"
    os.environ["HAI_JOB_REPORT"] = str(tmp / "report.txt")
    assert main() == 1  # bad.wav failed, the other was still done
    assert (tmp / "report.txt").read_text() == f"Meeting notes drafted: {note_name(root / 'a' / 'm [1].m4a')}\n"
    # Every transcription runs before any draft, and a failed one is not drafted.
    assert calls == ["bad.wav/t", "m [1].m4a/t", "m [1].m4a"], calls
    out = root / "me" / "m [1]"
    assert (out / "summary.md").read_text() == "draft" and (out / "m [1].txt").exists()
    # No date in "m [1]": named from the mtime, which the setup pinned to the epoch.
    epoch = note_name(root / "a" / "m [1].m4a")
    assert (tmp / "vault" / f"{epoch} draft.md").read_text() == "draft"
    assert (tmp / "vault" / f"{epoch} transcript.md").read_text() == "hello"
    assert note_name(Path("Sound record (2026-10-21 10_30_15).m4a")) == "10-21-26 10h30"
    assert note_name(Path("rec 2026-09-02T08:05.wav")) == "9-02-26 08h05"
    assert not (root / "me" / "bad" / "summary.md").exists()
    calls.clear()
    assert main() == 1 and calls == ["bad.wav/t"]  # finished one is skipped, failed one retried
    assert (root / "a" / "m [1].m4a").read_bytes() == b"x"  # inbox untouched
    # Own folder is excluded: on the recording box there is nothing to do.
    os.environ["HAI_HOSTNAME"] = "a"
    assert [p.name for p in pending(root, "a")] == ["own.m4a"]
    os.environ["HAI_HOSTNAME"] = "me"
    shutil.rmtree(tmp / "vault")
    shutil.rmtree(out)
    calls.clear()
    assert main() == 1 and (out / "summary.md").exists()  # missing vault: skipped, not failed
    shutil.rmtree(tmp)
    print("ok")


if __name__ == "__main__":
    selftest() if "--selftest" in sys.argv else sys.exit(main())
