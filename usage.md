# Použití programu


## Více videí najednou

```python
python -m src.process_video -i data -p "*.mkv" -o processed_videos/Taxol -n 5 
```

Zpracuje všechna mkv videa ve složce `data/`. Vytvoří a uloží je do `processed_videos/Taxol/` složky.
`n -5` použije 5 workerů.

## Jedno video

```python
python -m src.process_video -i data -p "C1.tuns.mkv" -o processed_videos/Control -n 5 
```

## Vybraná videa

Zpracuje T1.tuns.mkv, T2.tuns.mkv,... až T5.tuns.mkv.

```python
python -m src.process_video -i data -p "T[1-5].tuns.mkv" -o processed_videos/Taxol -n 5 
```
