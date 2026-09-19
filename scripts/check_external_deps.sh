#!/bin/bash
# External-code-dependency gate (Phase 6 Round 3, gate 3/3).
#
# Reads scripts/external_deps.txt (<path> <pinned-sha> <url>) and fails if any
# dependency is missing, drifted off its pinned commit, or locally modified.
# See the header of external_deps.txt for why this class of check exists.
#
# Usage:  bash scripts/check_external_deps.sh [--selftest]
# Exit:   0 = all pinned and clean, 1 = drift.
#         --selftest plants a wrong SHA in a scratch copy and must FAIL.
set -u
cd "$(git rev-parse --show-toplevel)" 2>/dev/null || exit 1

REG=scripts/external_deps.txt
SELFTEST=0
[ "${1:-}" = "--selftest" ] && SELFTEST=1

FAIL=0
ok ()  { echo "[OK]   $1"; }
bad () { echo "[FAIL] $1"; FAIL=$((FAIL+1)); }

check_one () {   # $1=registry file
  local reg="$1" n=0 bad_n=0 path sha url full have dirty
  echo "=============================================================="
  echo " external dependency check   (registry: $reg)"
  echo "=============================================================="
  if [ ! -f "$reg" ]; then bad "registry missing: $reg"; return 1; fi
  while read -r path sha url; do
    case "${path:-}" in ""|\#*) continue;; esac
    n=$((n+1))
    full="$path"
    if [ ! -d "$full" ]; then
      bad "missing dependency: $path  ($url)"; bad_n=$((bad_n+1)); continue
    fi
    have=$(git -C "$full" rev-parse HEAD 2>/dev/null || echo none)
    if [ "$have" != "$sha" ]; then
      bad "pinned-SHA drift: $path  have=$have  want=$sha"; bad_n=$((bad_n+1))
    fi
    dirty=$(git -C "$full" status --porcelain 2>/dev/null | head -1)
    if [ -n "$dirty" ]; then
      bad "dependency working tree DIRTY: $path  (e.g. $dirty)"; bad_n=$((bad_n+1))
    fi
    [ "$have" = "$sha" ] && [ -z "$dirty" ] && ok "pinned + clean: $path @ ${sha:0:12}"
  done < "$reg"
  if [ "$n" -eq 0 ]; then bad "registry contains no entries (nothing is being guarded)"; return 1; fi
  [ "$bad_n" -eq 0 ] && ok "all $n external dependency(ies) pinned and clean"
  return $([ "$bad_n" -eq 0 ] && echo 0 || echo 1)
}

if [ "$SELFTEST" -eq 1 ]; then
  TMP=$(mktemp -d)
  grep -v '^#' "$REG" | grep -v '^[[:space:]]*$' | head -1 > "$TMP/one"
  path=$(awk '{print $1}' "$TMP/one"); sha=$(awk '{print $2}' "$TMP/one"); url=$(awk '{print $3}' "$TMP/one")
  printf '%s %s %s\n' "$path" "0000000000000000000000000000000000000000" "$url" > "$TMP/bad.txt"
  echo "--- selftest: a deliberately wrong SHA must be caught ---"
  if check_one "$TMP/bad.txt" >/dev/null 2>&1; then
    echo "[FAIL] selftest: wrong SHA was NOT caught -- the gate is blind"
    rm -rf "$TMP"; exit 1
  fi
  echo "[OK]   selftest: wrong SHA correctly rejected"
  printf '%s %s %s\n' "$path" "$sha" "$url" > "$TMP/good.txt"
  if ! check_one "$TMP/good.txt" >/dev/null 2>&1; then
    echo "[FAIL] selftest: a correct entry was rejected -- the gate is over-strict"
    rm -rf "$TMP"; exit 1
  fi
  echo "[OK]   selftest: correct entry accepted"
  rm -rf "$TMP"
  echo
  echo "RESULT: external-deps selftest PASS"
  exit 0
fi

check_one "$REG"
echo
if [ "$FAIL" -eq 0 ]; then echo "RESULT: external dependencies OK"; else echo "RESULT: EXTERNAL DEPENDENCY DRIFT"; fi
exit $([ "$FAIL" -eq 0 ] && echo 0 || echo 1)
