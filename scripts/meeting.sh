#!/usr/bin/env bash
# Audio file -> diarized whisperX transcript -> meeting notes from a local LLM.
# Run with no arguments for usage.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)

usage() {
    cat >&2 <<EOF
usage: meeting.sh [options] <audio-file> [extra whisperx args...]

Audio file -> diarized whisperX transcript -> meeting notes from a local LLM.
Writes transcripts/<name>.{json,srt,txt,tsv,vtt} and
transcripts/<name>-notes/{part-NN.md,summary.md}.

Options (must come before the audio file):
  --copy-to DEST   also copy summary.md to DEST. A directory gets <name>.md,
                   a path ending in .md is used literally. The originals stay
                   in transcripts/ either way.
  --speakers MAP   rename diarization labels, comma-separated LABEL=Name pairs:
                   "SPEAKER_01=Alice,SPEAKER_02=Bob". Which label is who is
                   only visible in the transcript, so run once, skim
                   transcripts/<name>.txt, then re-run with the mapping.
  -h, --help       this message.

Environment:
  WHISPER_MODEL    whisper model (default: large-v2)
  BATCH_SIZE       whisperX batch size (default: 4; lower it if VRAM is tight)

Anything after the audio file goes to whisperX, e.g. --language fr.
Re-running skips transcription when the .json already exists, and skips any
chunk already written to transcripts/<name>-notes/. Delete those to force a
redo after changing the template.

Examples:
  scripts/meeting.sh "team sync 2026-07-30.m4a"
  scripts/meeting.sh --speakers "SPEAKER_01=Alice,SPEAKER_02=Bob" rec.m4a
  scripts/meeting.sh --copy-to ~/notes/meetings/ rec.m4a
  WHISPER_MODEL=large-v3 scripts/meeting.sh rec.wav --language fr
EOF
    exit "${1:-1}"
}

# if-form, not `[ ] && usage`: under set -e a false test there aborts the script.
if [ $# -eq 0 ]; then usage; fi

copy_to=""
speakers=""
while true; do
    case "${1:-}" in
        --copy-to)  copy_to=${2:?--copy-to needs a directory or .md path}; shift 2 ;;
        --speakers) speakers=${2:?--speakers needs LABEL=Name pairs}; shift 2 ;;
        -h|--help)  usage 0 ;;
        *) break ;;
    esac
done

audio=${1:-}
[ -n "$audio" ] || usage
shift

out=$repo/transcripts
name=$(basename "$audio")
json="$out/${name%.*}.json"

# whisperX pulls its own cuDNN/cuBLAS in as pip wheels, and nothing puts them on
# the linker path: without this the CUDA backend dies with
# "Unable to load any of {libcudnn_cnn.so.9...}" mid-run.
nv=$(echo "$repo"/.venv/lib/python*/site-packages/nvidia)
export LD_LIBRARY_PATH="$nv/cudnn/lib:$nv/cublas/lib:$nv/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

if [ -f "$json" ]; then
    echo "transcript already there, skipping whisperX: $json" >&2
else
    # A run that dies with no traceback is usually the OOM killer: earlyoom takes
    # this out around 3.5 GB RSS when free RAM runs low.
    "$repo/.venv/bin/whisperx" "$audio" \
        --model "${WHISPER_MODEL:-large-v2}" \
        --diarize \
        --batch_size "${BATCH_SIZE:-4}" \
        --compute_type float16 \
        --output_dir "$out" \
        --output_format all \
        "$@"
fi

# analyze_meeting.py streams progress to stderr and prints the summary path on stdout.
summary=$("$repo/.venv/bin/python" "$repo/scripts/analyze_meeting.py" "$json" \
    --template "$repo/scripts/meeting-template-simple.md" \
    --speakers "$speakers")

if [ -n "$copy_to" ]; then
    case "$copy_to" in
        *.md) dest=$copy_to ;;
        # Naming the copy after the meeting: a shared notes folder full of files
        # called summary.md would be useless.
        *) mkdir -p "$copy_to"; dest="${copy_to%/}/${name%.*}.md" ;;
    esac
    cp "$summary" "$dest"
    echo "$dest"
fi

echo "$summary"
