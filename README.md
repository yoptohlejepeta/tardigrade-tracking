# Tardigrade tracking 🌊🐻
```bash
uv run python -m src.process_video -i data/T5.tuns.mkv -o processed_videos/Taxol -n 5
```

> [!NOTE]
> ```bash
> ffmpeg -framerate 60 -pattern_type glob -i "images3/C2.tuns.mkv/*.png" -c:v libx264 -pix_fmt yuv420p output.mp4
> ```


> [!NOTE]
> copy all csv files
> ```bash
> rsync -avz -e "ssh -J pkotlan@10.12.0.5" --include='*/' --include='*.csv' --exclude='*' pkotlan@punta-gpus:/localdata/Scratch/zposelShare/CENAB/DATAImages/ .
> ```



## Object tracking algoritmy

- [Crocker-Grier](https://trackpy.readthedocs.io/en/stable/) 
- [Linear assignment problem](https://pemami4911.github.io/pdfs/Machine_Learning_for_Data_Association_in_Multi_Object_Tracking.pdf) 
- [Introduction to Assignment Methods in Tracking Systems](https://www.mathworks.com/help/fusion/ug/introduction-to-assignment-methods-in-tracking-systems.html) 





