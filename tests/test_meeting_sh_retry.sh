#!/bin/bash
# meeting.sh hears a failed float16 run again in int8, and only then: fake whisperx,
# python and nvidia-smi in a throwaway repo. Run: bash tests/test_meeting_sh_retry.sh
set -u
src=$(cd "$(dirname "$0")/.." && pwd)/scripts/meeting.sh
F=$(mktemp -d); trap 'rm -rf "$F"' EXIT
mkdir -p "$F/scripts" "$F/.venv/bin" "$F/bin" "$F/transcripts"; cp "$src" "$F/scripts/"
printf '#!/bin/bash\ncase "$*" in *torch*) echo 1;; *pick_whisper*) echo large-v3;; esac\n' > "$F/.venv/bin/python"
printf '#!/bin/bash\necho "$*" >> "$LOG"\nfor a in "$@"; do [ "$p" = --compute_type ] && ct=$a; [ "$p" = --output_dir ] && od=$a; p=$a; done\n[ "$ct" = "$FAIL_ON" ] && exit 1\necho "{}" > "$od/a.json"\n' > "$F/.venv/bin/whisperx"
printf '#!/bin/bash\ncase "$*" in *memory.free*) echo $FREE;; *) echo GPU0;; esac\n' > "$F/bin/nvidia-smi"
chmod +x "$F/.venv/bin/"* "$F/bin/"*; touch "$F/a.webm"
fails=0
# fail_on free expected_exit expected_runs
while read -r fail free want_exit want_runs; do
    : > "$F/log"; rm -f "$F/transcripts/a.json"
    LOG=$F/log FAIL_ON=$fail FREE=$free PATH=$F/bin:$PATH HOME=$F ASR=whisperx \
        bash "$F/scripts/meeting.sh" --transcribe-only "$F/a.webm" --languages nl,en >/dev/null 2>&1
    got=$?; runs=$(wc -l < "$F/log"); kept=$(grep -c -- '--languages nl,en' "$F/log")
    if [ "$got" != "$want_exit" ] || [ "$runs" != "$want_runs" ] || [ "$kept" != "$runs" ]; then
        echo "FAIL fail_on=$fail free=$free: exit $got runs $runs args kept $kept"; fails=1
    fi
done <<'CASES'
none 6000 0 1
float16 6000 0 2
int8 3000 1 1
CASES
[ "$fails" = 0 ] && echo ok
exit "$fails"
