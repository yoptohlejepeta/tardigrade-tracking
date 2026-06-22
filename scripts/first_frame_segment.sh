#!/usr/bin/env bash
# Segment only the first frame of each video in data/missing_C/ and save a PNG.
set -euo pipefail

INPUT_DIR="data/missing_C"
LABELS_DIR="labels/missing_C"
IMAGES_DIR="images"

for video in "$INPUT_DIR"/*.mkv; do
    name="$(basename "$video")"          # e.g. C4.tuns.mkv
    stem="${name%.mkv}"                  # e.g. C4.tuns

    echo ">>> Processing $name"

    # 1. Segment the first ~second of the video.
    uv run python -m src.extract_labels -i "$INPUT_DIR" -p "$name" -o "$LABELS_DIR" -n 4 -t 1 -s 60

    # 2. Keep only the first frame's labels.
    find "$LABELS_DIR/$stem" -name 'frame_*.npy' ! -name 'frame_000000.npy' -delete

    # 3. Visualize -> one PNG: $IMAGES_DIR/$stem/frame_000000.png
    uv run python -m src.track_and_visualize -i "$LABELS_DIR/$stem" -v "$video" -o "$IMAGES_DIR"
done

echo ">>> Done. PNGs are in $IMAGES_DIR/<video>/frame_000000.png"
