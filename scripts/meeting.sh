#!/usr/bin/env bash
# Audio file -> transcript (the home Tiiny, or diarized whisperX) -> meeting notes from an LLM.
# Run with no arguments for usage.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)

usage() {
    cat >&2 <<EOF
usage: meeting.sh [options] <audio-file> [extra whisperx args...]

Audio file -> transcript -> meeting notes from an LLM. On a machine with an
NVIDIA GPU the transcript is diarized whisperX on that GPU; elsewhere the home
Tiiny's Qwen3-ASR writes it through the HAI OS owner door, with no speaker
labels and nothing heavy on this machine's CPU. Writes transcripts/<name>.json and .txt (whisperX also
.srt, .tsv, .vtt) and
transcripts/<name>-notes/{part-NN.md,summary.md}.

Options (must come before the audio file):
  --copy-to DEST   also copy summary.md to DEST. A directory gets <name>.md,
                   a path ending in .md is used literally. The originals stay
                   in transcripts/ either way.
  --speakers MAP   (ASR=whisperx) rename diarization labels, comma-separated LABEL=Name pairs:
                   "SPEAKER_01=Alice,SPEAKER_02=Bob". Which label is who is
                   only visible in the transcript, so run once, skim
                   transcripts/<name>.txt, then re-run with the mapping.
                   Without it, voices enrolled with scripts/voices.py are
                   recognised and named automatically.
  --transcribe-only  stop after the transcript, before the LLM draft.
  -h, --help       this message.

Environment:
  ASR              whisperx (default where nvidia-smi sees a GPU) or tiiny
                   (default elsewhere: scripts/tiiny_transcribe.py, door URL in
                   ~/.config/hai/door)
  WHISPER_MODEL    whisper model (default: auto: on a GPU, large-v3-turbo when
                   the recording is English, --language en or heard so in its
                   first minutes, else large-v3; without a GPU large-v3-turbo)
  BATCH_SIZE       whisperX batch size (default: 4; lower it if VRAM is tight)
  WHISPER_HOTWORDS_FILE  words whisper should expect, such as product and
                   people names, comma-separated (default:
                   ~/.local/share/whisperx/hotwords.txt; skipped when absent)
  LLM_API          ollama (default), or openai for LM Studio and other
                   OpenAI-compatible servers
  LLM_HOST         default: localhost:11434 (ollama), localhost:1234/v1 (openai)
  LLM_MODEL        default: nemotron-3-nano:4b; required when LLM_API=openai

Anything after the audio file goes to whisperX, e.g. --language fr (the Tiiny
takes none). Re-running skips transcription when the .json already exists, and skips any
chunk already written to transcripts/<name>-notes/. Delete those to force a
redo after changing the template.

Examples:
  scripts/meeting.sh "team sync 2026-07-30.m4a"
  scripts/meeting.sh --speakers "SPEAKER_01=Alice,SPEAKER_02=Bob" rec.m4a
  scripts/meeting.sh --copy-to ~/notes/meetings/ rec.m4a
  WHISPER_MODEL=large-v3 scripts/meeting.sh rec.wav --language fr
  LLM_API=openai LLM_MODEL=google/gemma-4-26b-a4b-qat scripts/meeting.sh rec.m4a
EOF
    exit "${1:-1}"
}

# if-form, not `[ ] && usage`: under set -e a false test there aborts the script.
if [ $# -eq 0 ]; then usage; fi

copy_to=""
speakers=""
transcribe_only=""
while true; do
    case "${1:-}" in
        --copy-to)  copy_to=${2:?--copy-to needs a directory or .md path}; shift 2 ;;
        --speakers) speakers=${2:?--speakers needs LABEL=Name pairs}; shift 2 ;;
        --transcribe-only) transcribe_only=1; shift ;;
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

# whisperX on a CPU took 19 minutes for 150 s unfinished (2026-11-09): without a GPU the Tiiny hears.
if [ -z "${ASR:-}" ]; then
    if nvidia-smi -L >/dev/null 2>&1; then ASR=whisperx; else ASR=tiiny; fi
fi
if [ -f "$json" ]; then
    echo "transcript already there, skipping transcription: $json" >&2
elif [ "$ASR" = tiiny ]; then
    "$repo/.venv/bin/python" "$repo/scripts/tiiny_transcribe.py" "$audio" "$out" >/dev/null
else
    # On x86_64 the pinned torch is the CUDA build whether or not the machine has a
    # GPU, so the nvidia wheels being present proves nothing: ask torch instead.
    if [ "$("$repo/.venv/bin/python" -c 'import torch; print(int(torch.cuda.is_available()))')" = 1 ]; then
        # whisperX pulls its own cuDNN/cuBLAS in as pip wheels, and nothing puts them on
        # the linker path: without this the CUDA backend dies with
        # "Unable to load any of {libcudnn_cnn.so.9...}" mid-run.
        nv=$(echo "$repo"/.venv/lib/python*/site-packages/nvidia)
        export LD_LIBRARY_PATH="$nv/cudnn/lib:$nv/cublas/lib:$nv/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        compute=float16
        if [ "${WHISPER_MODEL:-auto}" = auto ]; then
            WHISPER_MODEL=$("$repo/.venv/bin/python" "$repo/scripts/pick_whisper.py" "$audio" "$@")
        fi
    else
        # float16 is a GPU format, and whisperX's CPU default of float32 costs hours on
        # a long recording for no accuracy worth having.
        compute=int8
    fi

    # Kept out of git: the list names this box's people and projects.
    hotwords_file=${WHISPER_HOTWORDS_FILE:-$HOME/.local/share/whisperx/hotwords.txt}
    hot=()
    if [ -s "$hotwords_file" ]; then hot=(--hotwords "$(tr '\n' ' ' < "$hotwords_file")"); fi

    # no GPU (ASR=whisperx forced): the cheapest model
    if [ "${WHISPER_MODEL:-auto}" = auto ]; then WHISPER_MODEL=large-v3-turbo; fi

    # A run that dies with no traceback is usually the OOM killer: earlyoom takes
    # this out around 3.5 GB RSS when free RAM runs low.
    "$repo/.venv/bin/whisperx" "$audio" \
        --model "$WHISPER_MODEL" \
        --diarize \
        --speaker_embeddings \
        --batch_size "${BATCH_SIZE:-4}" \
        --compute_type "$compute" \
        --output_dir "$out" \
        --output_format all \
        ${hot[@]+"${hot[@]}"} \
        "$@"
fi

if [ -n "$transcribe_only" ]; then echo "$json"; exit 0; fi

# No map given: name whoever voices.py recognises; the rest keep their SPEAKER_NN.
if [ -z "$speakers" ]; then
    speakers=$("$repo/.venv/bin/python" "$repo/scripts/voices.py" match "$json")
    echo "speakers recognised: ${speakers:-none}" >&2
fi

# if-form again: a false `[ ] && ...` here would abort the script under set -e.
llm=()
if [ -n "${LLM_API:-}" ];   then llm+=(--api "$LLM_API");   fi
if [ -n "${LLM_HOST:-}" ];  then llm+=(--host "$LLM_HOST"); fi
if [ -n "${LLM_MODEL:-}" ]; then llm+=(-m "$LLM_MODEL");    fi

# analyze_meeting.py streams progress to stderr and prints the summary path on stdout.
summary=$("$repo/.venv/bin/python" "$repo/scripts/analyze_meeting.py" "$json" \
    --template "$repo/scripts/meeting-template-simple.md" \
    --speakers "$speakers" \
    ${llm[@]+"${llm[@]}"})

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
