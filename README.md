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


## 20.10.

- spojovani stredu podle bboxu
- spojovani stredu podle max. vzdálenosti

- rozdělení posunů

- ( antimasky )

- 5 tuns, 5 Reh

## Object tracking algoritmy

- [Crocker-Grier](https://trackpy.readthedocs.io/en/stable/) 
- [Linear assignment problem](https://pemami4911.github.io/pdfs/Machine_Learning_for_Data_Association_in_Multi_Object_Tracking.pdf) 
- [Introduction to Assignment Methods in Tracking Systems](https://www.mathworks.com/help/fusion/ug/introduction-to-assignment-methods-in-tracking-systems.html) 

TODO

- cervena hranice


taxol od 1 a 5 (reh, tracking)


priklady snimku, spatne podminky (svetlo atd.)


---


- carovy graf pro kazdou skupinu (prumer ze vsech objektu)
- boxplot
- porovnani skupin (idealne Control Nocodazole)

pridat displacement, mean square displacement, area, zmena tvaru (feret)
aktivni/neaktivni

1. selekce snimku
2. jaka videa dohromady, seskupit podle OPUS.xlsx

jak spocitat mobility index (nebo podobnou metriku)


pozor na skupinu

objekty pres vsehna videa v dane skupine

1. vyfiltrovat objekty u kterých se mění indexy
2. analýza mobility - displacement (zkusit zatim jen tunsy, celkovy displacement)



suma delta r na druhou

stredni displacement

potom area (stejny postup)

pustit C8.tuns, C15.tuns
jeste dve videa z nocodazolu


- dodelat nocodazole


## Revize

- najít problémy
    1. analýza toho jak segmentovat reh videa (adaptace watershedu?)
    2. tracking zelvusek
    3. analyza na GPU

