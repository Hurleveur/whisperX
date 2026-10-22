# How to get a meeting transcribed and drafted without starting anything

Drop an audio file in your box's recordings folder. Another box that shares the
folder (over Syncthing) transcribes it and drafts the notes; `scripts/meeting_inbox.py`
does this when the hai-os job runner calls it.

1. **Drop the file** in `<hai-os>/.hai-shared/recordings/<your hostname>/`.
   Accepted: `.m4a .wav .mp3 .ogg .opus .flac .aac`. A file under two minutes old is
   skipped until the next run, so a recording still being written is left alone.
2. **Wait.** The job runs on a box whose hostname differs from the folder's: it treats
   every other host's folder as its inbox and never its own. It runs only while the
   box is idle and on mains power, and takes up to an hour per recording.
3. **Find the result** in `recordings/<processing box>/<recording name>/`: the whisperX
   files (`.json .txt .srt .vtt .tsv`) and `summary.md`, the draft. `summary.md` is
   written last; a folder without it is unfinished and the recording is retried.
   If `$MEETINGS_VAULT_DIR` (default `~/Documents/GitHub/sandbox-office-vault/team/alexandre/meetings`)
   exists, `<M-DD-YY> <HH>h<MM> draft.md` and `... transcript.md` are copied there too,
   dated from the recording's file name (or its modification time).
4. **Name the speakers and redo the draft.** Voices enrolled earlier are named
   automatically (step 5). For anyone else, read the `.txt`, note which `SPEAKER_NN`
   is who, then on the processing box, from the whisperX checkout:

       rm -r transcripts/<name>-notes
       scripts/meeting.sh --speakers "SPEAKER_00=Alice,SPEAKER_01=Bob" <the audio file>

   The transcript is reused, only the draft is regenerated into `transcripts/`. Copy
   `transcripts/<name>-notes/summary.md` over `recordings/<box>/<name>/summary.md`.
5. **Enroll a voice once**, from any transcript where you know who `SPEAKER_NN` is,
   with the person's agreement (a voice embedding identifies them):

       .venv/bin/python scripts/voices.py enroll transcripts/<name>.json SPEAKER_01 Bob

   The voices stay on the box that enrolled them, in `~/.local/share/whisperx/voices`
   (`VOICES_DIR` to change it), and never sync: enroll on each box that runs the job.

   From then on a label at least `VOICE_THRESHOLD` (default 0.7) similar to an
   enrolled voice gets that name; `scripts/voices.py match <json>` prints the scores.
   Enrolling the same person from a second recording adds a sample and helps when the
   room or microphone differs. Transcripts made before `--speaker_embeddings` was
   added to meeting.sh carry no embeddings: re-run whisperX on them to enroll.
6. **Run it by hand:** `HAI_RECORDINGS=/path/to/recordings .venv/bin/python scripts/meeting_inbox.py`.
   Exit code 1 and a `FAILED <file>: <reason>` line on stderr mean a file failed; the
   others were still processed. To retry a finished one, delete its output folder.

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
