#!/usr/bin/env bash
# Apply a release tarball over this working tree.
#
# Picks the NEWEST matching tarball rather than a fixed filename, because
# browsers save repeat downloads as "...(1).tar.gz" and updating from a stale
# file has already cost several debugging rounds.
#
# Leaves data/ and artifacts/ untouched.
#
#   ./update.sh                 # newest in ~/Downloads
#   ./update.sh /path/to.tar.gz # explicit
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

TMP=$(mktemp -d); trap 'rm -rf "$TMP"' EXIT
tar xzf "$SRC" -C "$TMP"
NEW="$TMP/wsi-metastasis-seg"
[[ -d "$NEW" ]] || { echo "unexpected tarball layout:"; ls "$TMP"; exit 1; }

echo
echo "--- incoming version (before copying anything) ---"
cat "$NEW/VERSION" 2>/dev/null || echo "no VERSION file"
echo "--- current version ---"
cat "$HERE/VERSION" 2>/dev/null || echo "no VERSION file"
echo

for item in src scripts configs docs tests VERSION Makefile pyproject.toml \
            README.md .importlinter update.sh; do
  [[ -e "$NEW/$item" ]] && cp -r "$NEW/$item" "$HERE/"
done

echo "--- verifying ---"
cd "$HERE"
python scripts/version.py
