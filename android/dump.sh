#!/usr/bin/env bash
# Usage: ./dump.sh [tag] [n] [proc]
#   tag   name appended to the dump directory (default: manual)
#   n     number of consecutive frames (default: 30)
#   proc  off | default  (optional; switches the processing mode first)
# Triggers a frame dump on the phone and pulls it into <repo>/captures/.
set -u
PKG=me.stedra.stegocam
TAG="${1:-manual}"
N="${2:-30}"
PROC="${3:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="$HERE/../captures"
mkdir -p "$OUT"

adb shell input keyevent KEYCODE_WAKEUP >/dev/null 2>&1
# only (re)start if the app is not already the resumed activity (avoids cycling the camera)
if adb shell dumpsys activity activities 2>/dev/null | grep -E "ResumedActivity" | grep -q "$PKG/"; then
  sleep 0.5
else
  adb shell am start -n "$PKG/.MainActivity" >/dev/null 2>&1
  sleep 2
fi

adb logcat -c
LOG="$(mktemp)"
trap 'kill "${LPID:-0}" 2>/dev/null; wait "${LPID:-0}" 2>/dev/null; rm -f "$LOG"' EXIT
adb logcat -v brief -s StegoCam:I > "$LOG" 2>/dev/null &
LPID=$!
sleep 0.5

if [[ -n "$PROC" ]]; then
  adb shell am broadcast -a $PKG.DUMP -p $PKG --ei n "$N" --es tag "$TAG" --es proc "$PROC" >/dev/null
else
  adb shell am broadcast -a $PKG.DUMP -p $PKG --ei n "$N" --es tag "$TAG" >/dev/null
fi

REMOTE=""
for _ in $(seq 1 600); do   # 60 s
  if grep -q 'StegoCam DUMP_FAILED' "$LOG"; then
    echo "dump failed: $(grep -m1 'StegoCam DUMP_FAILED' "$LOG" | tr -d '\r')" >&2
    exit 1
  fi
  line="$(grep -m1 'StegoCam DUMPED' "$LOG" | tr -d '\r')"
  if [[ -n "$line" ]]; then REMOTE="${line##*StegoCam DUMPED }"; break; fi
  sleep 0.1
done
if [[ -z "$REMOTE" ]]; then
  echo "timeout: no 'StegoCam DUMPED' line within 60 s (screen on/unlocked? camera permission granted?)" >&2
  exit 1
fi

adb pull "$REMOTE" "$OUT/" >/dev/null 2>&1 || { echo "adb pull failed for $REMOTE" >&2; exit 1; }
echo "$OUT/$(basename "$REMOTE")"
