#!/usr/bin/env bash
# Build ONE .zip Kaggle will accept, from the complete patchset on disk.
#
# Kaggle limits a dataset to 50 top-level files and auto-expands .tar archives
# into individual files. 116,069 patches = 232,140 files, ~4,600x the limit,
# so a loose upload is truncated (BUG: only ~12% landed). A single .zip is one
# file, and Kaggle leaves .zip intact. The notebook unzips it at runtime.
set -euo pipefail

PS="${1:-artifacts/patchset}"          # path to your exported patchset
OUT="${2:-kaggle_zip}"

[[ -d "$PS/shards" ]] || { echo "no $PS/shards -- pass the patchset dir as arg 1"; exit 1; }
n=$(ls "$PS"/shards/*.tar | wc -l)
echo "zipping $n shards + manifest from $PS"

rm -rf "$OUT"; mkdir -p "$OUT"
# zip stores the shards/ tree and the two small files inside one archive
( cd "$PS" && zip -r -0 "$OLDPWD/$OUT/patchset.zip" shards manifest.parquet export_info.json )
#  -0 = store, no compression: the shards are already JPEG, so compressing
#       wastes minutes for ~0% gain.

sz=$(du -h "$OUT/patchset.zip" | cut -f1)
echo "wrote $OUT/patchset.zip ($sz)"

cat > "$OUT/dataset-metadata.json" <<JSON
{
  "title": "CAMELYON16 patchset 512 zip",
  "id": "archittiwari99/camelyon16-patchset-512-zip",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON
echo
echo "Now upload the ONE zip. Faster and resumable-in-practice from Kaggle:"
echo "  put patchset.zip on Drive, then in a Kaggle cell:"
echo "     gdown the zip -> kaggle datasets create -p <dir>"
echo "Or directly if your uplink cooperates:"
echo "     kaggle datasets create -p $OUT"
