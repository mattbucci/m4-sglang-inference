#!/bin/bash
# test_patch_gates.sh — 3-gate patch-hygiene test (fleet protocol; 3090 donor).
# Scripted so every rebase / new-patch commit runs the same gates instead of
# re-implementing them by hand.
#
# Gates:
#   (a) every patch in PATCH_DIR applies clean, in setup.sh's glob order, on a
#       PRISTINE $SGLANG_TAG worktree (no skips — a skip on pristine means a
#       broken chain that an idempotent check-apply-else-skip loop would hide);
#   (b) the patched pristine worktree is byte-identical to the live SGLANG_DIR
#       tree (catches live-tree edits that never made it into a patch — the
#       patch-013 fabrication class);
#   (c) every patch FAILS `git apply --check` on the live (already-patched)
#       tree (rerun safety — a patch that still applies twice is mis-anchored).
#
# Usage:
#   scripts/test_patch_gates.sh                     # gates for the live stack
#   SGLANG_TAG=<tag> SGLANG_DIR=<tree> PATCH_DIR=<dir> \
#     scripts/test_patch_gates.sh                   # scratch stack / bisect arm
#
# Exit: 0 all gates pass; 1 gate failure; 2 environment error (missing dirs /
# tag object absent — distinct from a clean GATE-A FAIL).
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

PATCH_DIR="${PATCH_DIR:-$REPO/patches}"
case "$PATCH_DIR" in /*) ;; *) PATCH_DIR="$REPO/$PATCH_DIR" ;; esac
[ -d "$PATCH_DIR" ] || { echo "FATAL: PATCH_DIR $PATCH_DIR missing"; exit 2; }
SGLANG_DIR="${SGLANG_DIR:-$REPO/components/sglang}"
[ -d "$SGLANG_DIR/.git" ] || [ -f "$SGLANG_DIR/.git" ] \
  || { echo "FATAL: SGLANG_DIR $SGLANG_DIR is not a git tree"; exit 2; }
SGLANG_TAG="${SGLANG_TAG:-v0.5.15.post1}"

# setup.sh's exact apply glob — the gate must test the chain setup.sh applies.
PATCHES=("$PATCH_DIR"/0[01][0-9]-*.patch)
[ -e "${PATCHES[0]}" ] || { echo "FATAL: no patches match $PATCH_DIR/0[01][0-9]-*.patch"; exit 2; }
N=${#PATCHES[@]}
WT_PARENT="$(mktemp -d "${TMPDIR:-/tmp}/patch-gates-XXXXXX")"
WT="$WT_PARENT/wt"
FAIL=0
echo "== 3-gate patch test: $N patches from $PATCH_DIR vs $SGLANG_TAG =="

# --- gate (a): glob-order apply on pristine tag worktree ---
git -C "$SGLANG_DIR" worktree add -f "$WT" "$SGLANG_TAG" >/dev/null 2>&1 \
  || { echo "FATAL: cannot create worktree at $SGLANG_TAG (tag object absent? shallow clone holds only the pinned tag)"; exit 2; }
A_OK=0
for p in "${PATCHES[@]}"; do
  if git -C "$WT" apply "$p" 2>/dev/null; then A_OK=$((A_OK+1));
  else echo "  GATE-A FAIL (does not apply on pristine): $(basename "$p")"; FAIL=1; fi
done
echo "gate (a): $A_OK/$N apply clean on pristine $SGLANG_TAG"

# --- gate (b): byte-identity vs live tree ---
DIFFS=$(diff -rq "$WT/python/sglang" "$SGLANG_DIR/python/sglang" 2>&1 \
  | grep -v "_version.py\|egg-info\|__pycache__" || true)
if [ -n "$DIFFS" ]; then
  echo "gate (b): FAIL — live tree differs from pristine+patches:"; echo "$DIFFS" | head -10; FAIL=1
else
  echo "gate (b): live tree byte-identical to pristine+patches"
fi

# --- gate (c): rerun safety on the live tree ---
C_OK=0
for p in "${PATCHES[@]}"; do
  if git -C "$SGLANG_DIR" apply --check "$p" 2>/dev/null; then
    echo "  GATE-C FAIL (still applies on patched tree): $(basename "$p")"; FAIL=1
  else C_OK=$((C_OK+1)); fi
done
echo "gate (c): $C_OK/$N correctly fail on the patched tree"

git -C "$SGLANG_DIR" worktree remove --force "$WT" >/dev/null 2>&1
rm -rf "$WT_PARENT" 2>/dev/null
[ "$FAIL" = 0 ] && echo "== ALL GATES PASS ==" || echo "== GATE FAILURES (see above) =="
exit $FAIL
