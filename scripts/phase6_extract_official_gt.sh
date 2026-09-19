#!/usr/bin/env bash
# Phase 6 -- stage the OFFICIAL BDD100K annotation packages so the third-party
# pipelines (TriLiteNet / TwinLiteNet / TwinLiteNet+) can be run *unmodified*.
#
# Source zips (links in the upstream READMEs):
#   ~/Downloads/{det,da_seg,ll_seg}_annotations.zip
# Target layout (sibling of images/, which is how their code finds it):
#   data/bdd100k/official_eval/
#     images/val/<id>.jpg                       (symlink -> ../images/100k/val)
#     det_annotations/val/<id>.json
#     drivable_are_annotations/val/<id>.png
#     lane_line_annotations/val/<id>.png
#
# Only the `val` split is unpacked: that is the 10,000-image set our `tri_val`
# evaluation uses, and the only split an evaluation ever reads. `train` is left
# in the zip -- we never train these baselines, we load their released weights.
#
# Their loaders resolve labels by STRING REPLACEMENT on the image path
# (TwinLiteNetPlus/BDD100K.py:194,196), so the directory names matter.
set -u
cd "$(dirname "$0")/.."
SRC=${SRC:-/mnt/c/Users/<user>/Downloads}
D=data/bdd100k/official_eval
mkdir -p "$D"
[ -e "$D/images" ] || ln -sfn ../images/100k "$D/images"

stage() {  # $1 = zip name, $2 = destination dir
  local zip="$1" dst="$2"
  if [ ! -f "$SRC/$zip" ]; then echo "!! missing $SRC/$zip"; return 1; fi
  echo "=== $zip -> $dst/val ==="
  mkdir -p "$D/$dst/val"
  unzip -j -o -q "$SRC/$zip" "*/val/*" -d "$D/$dst/val"
  printf '  val files: %s\n' "$(ls "$D/$dst/val" 2>/dev/null | wc -l)"
}

stage det_annotations.zip     det_annotations
stage da_seg_annotations.zip  drivable_are_annotations
stage ll_seg_annotations.zip  lane_line_annotations

echo "=== layout ==="
find "$D" -maxdepth 2 -mindepth 1 | sort | while read -r p; do
  if [ -d "$p" ]; then printf '  %-52s %s files\n' "$p" "$(ls "$p" 2>/dev/null | wc -l)"
  else printf '  %-52s -> %s\n' "$p" "$(readlink "$p" 2>/dev/null || echo file)"; fi
done
echo "STAGE DONE"
