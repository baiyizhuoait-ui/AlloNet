#!/bin/bash
# Download third-party baseline pretrained weights into weights/.
# These binaries are subject to their own licenses; they are NOT part of this repo.
set -e
mkdir -p weights
cd weights

# YOLOP (hustvl/YOLOP) - end-to-end multitask network
if [ ! -f YOLOP_End-to-end.pth ]; then
  echo "Download YOLOP end-to-end weights from https://github.com/hustvl/YOLOP (see its README/release)"
  curl -L -o YOLOP_End-to-end.pth "https://github.com/hustvl/YOLOP/releases/download/v1.0/yolop_end_to_end.pth" || echo "Please fetch manually from the YOLOP repo"
fi

# TwinLiteNetPlus (duyanh2604/TwinLiteNetPlus)
for v in small medium large; do
  if [ ! -f TwinLiteNetPlus_${v}.pth ]; then
    echo "Download TwinLiteNetPlus-${v} from https://github.com/duyanh2604/TwinLiteNetPlus (see its README)"
  fi
done

# TriLiteNet (duyanh2604/TriLiteNet)
for v in nano small base; do
  if [ ! -f TriLiteNet_${v}.pth ]; then
    echo "Download TriLiteNet-${v} from https://github.com/duyanh2604/TriLiteNet (see its README)"
  fi
done

echo "Done. Note: upstream URLs may change; check each official repo if a download fails."
