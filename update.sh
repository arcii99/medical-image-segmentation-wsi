#!/usr/bin/env bash
# Apply a release tarball over this working tree.
#
# Picks the NEWEST matching tarball rather than a fixed filename, because
# browsers save repeat downloads as "...(1).tar.gz" and updating from a stale
# file has already cost several debugging rounds.
#
# Leaves data/ and artifacts/ untouched.
#
# Invoke with `bash update.sh` rather than `./update.sh`. The execute bit does
# not survive every download/extract path reliably, and "Permission denied"
# has already cost one round here.
#
#   bash update.sh                 # newest in ~/Downloads
#   bash update.sh /path/to.tar.gz # explicit
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${1:-}"
if [[ -z "$SRC" ]]; then
  SRC=$(ls -t "$HOME"/Downloads/wsi-metastasis-seg*.tar.gz 2>/dev/null | head -1 || true)
fi
[[ -n "$SRC" && -f "$SRC" ]] || { echo "no tarball found; pass one explicitly"; exit 1; }

echo "tarball : $SRC"
echo "          $(date -r "$SRC" '+%Y-%m-%d %H:%M')  $(du -h "$SRC" | cut -f1)"
echo "target  : $HERE"

# Show every candidate. Browsers save repeat downloads as "...(1).tar.gz",
# and a stale pick is the single most common way an update silently no-ops.
CANDIDATES=$(ls -t "$HOME"/Downloads/wsi-metastasis-seg*.tar.gz 2>/dev/null || true)
if [[ $(echo "$CANDIDATES" | wc -l) -gt 1 ]]; then
  echo
  echo "other candidates in ~/Downloads (newest first):"
  while read -r f; do
    [[ -n "$f" ]] && printf "  %s  %s\n" "$(date -r "$f" '+%Y-%m-%d %H:%M')" "$f"
  done <<< "$CANDIDATES"
fi

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
tar xzf "$SRC" -C "$TMP"
NEW="$TMP/wsi-metastasis-seg"
[[ -d "$NEW" ]] || { echo "unexpected tarball layout:"; ls "$TMP"; exit 1; }

INCOMING=$(cat "$NEW/VERSION" 2>/dev/null || echo "unknown")
CURRENT=$(cat "$HERE/VERSION" 2>/dev/null || echo "unknown")
echo
echo "incoming version : $INCOMING"
echo "current  version : $CURRENT"

if [[ "$INCOMING" == "$CURRENT" ]]; then
  echo
  echo "=============================================================="
  echo " NOTHING TO DO -- the tarball is the version already installed."
  echo ""
  echo " This almost always means the download is STALE: the newer"
  echo " release was never saved to ~/Downloads. Check the timestamp"
  echo " above against when the release was published."
  echo ""
  echo " Download the current tarball, then re-run this script."
  echo " Pass an explicit path if it landed somewhere else:"
  echo "     bash update.sh /path/to/wsi-metastasis-seg.tar.gz"
  echo "=============================================================="
  exit 2
fi
echo

for item in src scripts configs docs tests notebooks VERSION Makefile \
            pyproject.toml README.md .importlinter; do
  [[ -e "$NEW/$item" ]] && cp -r "$NEW/$item" "$HERE/"
done

# update.sh is handled separately and LAST, via a rename rather than a copy.
#
# bash reads a script incrementally by byte offset. `cp` truncates and
# rewrites the same inode, so overwriting this file mid-run makes bash resume
# at the old offset inside the NEW contents -- it lands mid-line and fails
# with something unrelated to the real problem ("f: unbound variable", in the
# case that prompted this comment).
#
# `mv` swaps the directory entry for a different inode. The running process
# keeps reading the original file, which stays intact until it exits.
if [[ -e "$NEW/update.sh" ]]; then
  cp "$NEW/update.sh" "$HERE/.update.sh.incoming"
  chmod +x "$HERE/.update.sh.incoming" 2>/dev/null || true
  mv -f "$HERE/.update.sh.incoming" "$HERE/update.sh"
fi

echo "--- verifying ---"
cd "$HERE"
python scripts/version.py
