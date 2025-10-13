# Tardigrade tracking 🌊🐻
```
uv run python -m src.process_video -i data/T5.tuns.mkv -o processed_videos/Taxol -n 5
```

> [!NOTE]
> `ffmpeg -framerate 60 -pattern_type glob -i "images3/C2.tuns.mkv/*.png" -c:v libx264 -pix_fmt yuv420p output.mp4`

## TODO

popsat clenitost

staticke
- plocha objektu
- max/min feret polomer
- obvod a jeho clenitost
- solidity
- sfericita
- excentricity
- cirkularita
- aproximace objektu elipsou (velka + mala poloosa)

dynamicke
- delka trajektorie
- displacement
- směrovost pohybu

id snimku, id objektu

poresit labelovani

tuns = vysychani
reh = rehydrated

3 latky

1. zlepsit watershed
2. jak se meni pocet clusteru
3. jak dlouho trva video, analyzovat video
4. pripravit parametry (zatim netreba pocitat)
5. teselace (nemusi byt do priste), voroneho diagram

zaridit pristu na pocitac

kazdy snimek do jedne slozky!


## prezentace

tuns snimek -> workflow segmentace -> pocty objektu (i v prubehu casu) -> seznam parametru + vykreslit

tuns -> pocet objektu + graf pocet objektu pro reh

pro dvojice tun a reh ( grafy clusteru )

### grafy

- pridat grafy distribuco


## 6.10.

- c2, c3 reh
- watershed
