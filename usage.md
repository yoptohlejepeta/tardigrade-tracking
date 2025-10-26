# Použití programu

## Segmentace

### Více videí najednou

```python
python -m src.extract_labels -i data/Control -p "*.mkv" -o labels/Control -n 5
```

Zpracuje všechna mkv videa ve složce `data/`. Vytvoří a uloží je do `labels/Control/` složky.
`n -5` použije 5 workerů.

### Jedno video

```python
python -m src.extract_labels -i data/Control -p "C1.tuns.mkv" -o labels/Control -n 5 
```

### Vybraná videa

Zpracuje T1.tuns.mkv, T2.tuns.mkv,... až T5.tuns.mkv.

```python
python -m src.extract_labels -i data -p "T[1-5].tuns.mkv" -o labels/Taxol -n 5 
```


## Tracking

TODO ...


## video

```bash
ffmpeg -framerate 60 -pattern_type glob -i "images/Control/C7.tuns/*.png" -c:v libx264 -pix_fmt yuv420p C7.tuns.mp4
```
