#!/bin/bash

for NUM in {1..3}
do
  ffmpeg -framerate 60 -pattern_type glob -i "processed_videos/Taxol/Reh/T${NUM}.72Reh.mkv/*.png" -c:v libx264 -pix_fmt yuv420p "video_outputs/Taxol/Reh/T${NUM}.72Reh.mkv"
done
