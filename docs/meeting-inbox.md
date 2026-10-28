# How to get a meeting transcribed and drafted without starting anything

Drop an audio file in a meetings folder you chose. The box that watches it transcribes
it, drafts the notes beside it, and moves the audio out to an archive;
`scripts/meeting_inbox.py` does this when the hai-os job runner calls it.

1. **Choose the folder, on one box.** Set `MEETINGS_INBOX` to the folder's path in the
   job's environment. Under hai-os that is this machine's `background/jobs.local.toml`
   (gitignored, never synced). The cockpit's Background panel sets it: type the path in
   the **folder** box on the `meeting-inbox` row. By hand, the file reads:

       [jobs.meeting-inbox.env]
       MEETINGS_INBOX = "~/path/to/the/meetings/folder"

   Unset, the job does nothing. Set it on one box only: a folder shared by sync
   (Syncthing, a Solid pod) is seen by every box, and two boxes would each take the
   same recording. The setting is read on every run, so nothing needs restarting.
2. **Drop the file** in that folder. Accepted: `.m4a .wav .mp3 .mp4 .webm .ogg .opus
   .flac .aac`. A file under two minutes old is skipped until the next run, so a
   recording still being written or synced in is left alone. The file name may be the
   same every time (a recorder that always saves `Recording.m4a`): a name without a
   date and time (`2026-10-21 10_30`, `2026-10-21T10:30`) is renamed in the folder with
   its modification time when the job picks it up, `Recording 2026-10-21 10_30.m4a`, so
   each recording gets its own transcript and archive folder. A name already taken in
   the folder or the archive gets a number.
3. **Wait.** The job runs only while the box is idle and on mains power, and takes up
   to an hour per recording.
4. **Find the result.** `<M-DD-YY> <HH>h<MM> draft.md` and `... transcript.md` appear in
   the meetings folder, dated from the recording's file name (or its modification
   time); a name already taken gets a number, so a note is never written over. The
   audio leaves the folder: it is moved, with the whisperX files (`.json .txt .srt .vtt
   .tsv`) and `summary.md` (the draft as made), to the archive
   `$HAI_RECORDINGS/<this box's hostname>/<recording name>/` (default
   `~/Documents/GitHub/hai-os/.hai-shared/recordings`). A recording still in the folder
   is unfinished and is retried; one whose archive already holds `summary.md` is only
   moved, never redrafted.

   **A note already written for that day** (`<M-DD-YY>.md`, the team's own name for a
   meeting note) is not given a draft beside it: the model reads that note first, then
   the draft, and writes the note back with what the recording adds. The merge is kept
   only if every line the team wrote is still there, unchanged and in order, and nobody
   edited the note while the model ran; the note as it was is copied to the archive
   first, as `<M-DD-YY> before merge.md`. A merge that fails either check is thrown
   away and the draft is written beside the note as usual. The model chooses where the
   additions go (on the first live runs it added a section at the end), and its
   additions are not checked against the transcript beyond the draft's own quotes:
   read them as a draft. The merge uses the same model and cloud purpose as the draft,
   so the team's note leaves the box for that call too.
5. **Name the speakers and redo the draft.** Voices enrolled earlier are named
   automatically (step 6). For anyone else, read the `.txt`, note which `SPEAKER_NN`
   is who, then on the processing box, from the whisperX checkout:

       rm -r transcripts/<name>-notes
       scripts/meeting.sh --speakers "SPEAKER_00=Alice,SPEAKER_01=Bob" <the archived audio file>

   The transcript is reused, only the draft is regenerated into `transcripts/`. Copy
   `transcripts/<name>-notes/summary.md` over the draft in the meetings folder.
6. **Enroll a voice once**, from any transcript where you know who `SPEAKER_NN` is,
   with the person's agreement (a voice embedding identifies them):

       .venv/bin/python scripts/voices.py enroll transcripts/<name>.json SPEAKER_01 Bob

   The voices stay on the box that enrolled them, in `~/.local/share/whisperx/voices`
   (`VOICES_DIR` to change it), and never sync: enroll on each box that runs the job.

   From then on a label at least `VOICE_THRESHOLD` (default 0.7) similar to an
   enrolled voice gets that name; `scripts/voices.py match <json>` prints the scores.
   Enrolling the same person from a second recording adds a sample and helps when the
   room or microphone differs. Transcripts made before `--speaker_embeddings` was
   added to meeting.sh carry no embeddings: re-run whisperX on them to enroll.
7. **Run it by hand:** `MEETINGS_INBOX=/path/to/meetings .venv/bin/python scripts/meeting_inbox.py`.
   Exit code 1 and a `FAILED <file>: <reason>` line on stderr mean a file failed; the
   others were still processed. To redo a finished one, move its audio back into the
   folder and delete its archive folder.

Decisions, action items and open questions in the draft each end with the words that
settled or raised them, in quotes. `analyze_meeting.py` deletes any such bullet whose
quote is not in the transcript, so a decision the model inferred never reaches the
notes. An action item also opens with its owner as a tag (`#Alice`), and the quote must
be words that owner said; one without an owner, or quoting someone else, is deleted.
A last section, `Said agreements`, is plain code: every transcript line voicing an
agreement ("je suis d'accord", "on s'est mis d'accord", "agreed"), with the two lines
before it, verbatim. It finds what the model loses and invents nothing.

A quote can be real and still not support the claim beside it, and the summary prose is
not checked at all: read the draft as a rough index of the transcript, not as the record.

## What the processing box needs

- This checkout with `.venv` (`uv sync`), and the `large-v3-turbo` weights (downloaded on first use).
- A Hugging Face token the pyannote diarization models accept, as `HF_TOKEN` in the
  job's environment or in the `huggingface_hub` login cache, with the gated model terms accepted.
- hai-context running on `127.0.0.1:18701` and a bearer token in `~/.config/hai/tokens/scheduler`
  (`LLM_TOKEN_FILE` overrides the path). Drafting goes out on hai-context's `meeting` cloud
  purpose (Gemma 4 31B on a zero-data-retention OpenRouter endpoint, metered under your
  caps, about $0.02 for an hour of meeting): the transcript leaves the box for that call.
  `LLM_ESCALATE=` (empty) keeps it on the local model `hai-local` instead.
- When the job finishes a recording it leaves a quiet row in the cockpit's Activity
  ("Meeting notes drafted: 10-21-26 11h30"), with no popup on the avatar.
- `WHISPER_MODEL`, `LLM_API`, `LLM_HOST`, `LLM_MODEL` in the environment override the defaults.
- Optionally, the names whisper keeps mishearing (people, products, project names),
  comma-separated in `~/.local/share/whisperx/hotwords.txt` (`WHISPER_HOTWORDS_FILE`
  overrides the path). It stays on the box, like the voices.
