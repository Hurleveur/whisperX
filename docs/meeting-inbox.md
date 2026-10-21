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
   exists, `<name> draft.md` and `<name> transcript.txt.md` are copied there too.
4. **Name the speakers and redo the draft.** Read the `.txt`, note which `SPEAKER_NN`
   is who, then on the processing box, from the whisperX checkout:

       rm -r transcripts/<name>-notes
       scripts/meeting.sh --speakers "SPEAKER_00=Alice,SPEAKER_01=Bob" <the audio file>

   The transcript is reused, only the draft is regenerated into `transcripts/`. Copy
   `transcripts/<name>-notes/summary.md` over `recordings/<box>/<name>/summary.md`.
5. **Run it by hand:** `HAI_RECORDINGS=/path/to/recordings .venv/bin/python scripts/meeting_inbox.py`.
   Exit code 1 and a `FAILED <file>: <reason>` line on stderr mean a file failed; the
   others were still processed. To retry a finished one, delete its output folder.

## What the processing box needs

- This checkout with `.venv` (`uv sync`), and the `large-v3-turbo` weights (downloaded on first use).
- A Hugging Face token the pyannote diarization models accept, as `HF_TOKEN` in the
  job's environment or in the `huggingface_hub` login cache, with the gated model terms accepted.
- hai-context running on `127.0.0.1:18701` and a bearer token in `~/.config/hai/tokens/scheduler`
  (`LLM_TOKEN_FILE` overrides the path). Drafting uses the local model `hai-local` only.
- `WHISPER_MODEL`, `LLM_API`, `LLM_HOST`, `LLM_MODEL` in the environment override the defaults.
