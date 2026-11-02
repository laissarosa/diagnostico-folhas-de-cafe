# src/check_sources.py
from collections import Counter
from pathlib import Path
import pandas as pd
from PIL import Image
from config import ROOT

M = ROOT / "data" / "raw" / "bracol_manual"
Y = ROOT / "data" / "raw" / "BRACOL-ORIGINAL-ANNOTATIONS" / "BRACOL-ORIGINAL-DETECT"
NAMES = {0: "healthy", 1: "miner", 2: "rust", 3: "phoma", 4: "cercospora"}
EXT = {".jpg", ".jpeg", ".png"}

csv = next(M.rglob("dataset.csv"))
img_dir = max((d for d in M.rglob("images") if d.is_dir()),
              key=lambda d: sum(1 for _ in d.iterdir()))

df = pd.read_csv(csv).dropna(subset=["id", "predominant_stress"]).copy()
df["id"] = df["id"].astype(int)
df["label"] = df["predominant_stress"].astype(int).map(NAMES)
have = {p.stem for p in img_dir.iterdir() if p.is_file()}
df["tem_imagem"] = df["id"].astype(str).isin(have)

print("Linhas no CSV:", len(df), "| imagens no disco:", len(have))
print(pd.crosstab(df["label"], df["tem_imagem"])
        .rename(columns={True: "COM imagem", False: "SEM imagem"}).to_string())


def sizes(paths, n=40):
    paths = sorted(paths)
    paths = paths[::max(1, len(paths) // n)]
    c = Counter()
    for p in paths:
        with Image.open(p) as im:
            c[(im.size, im.format)] += 1
    return c.most_common(3)


yolo = [p for s in ("train", "valid", "test")
        for p in (Y / s / "images").glob("*") if p.suffix.lower() in EXT]
print("\nTamanho (largura, altura) e formato, amostra:")
print("Mendeley:", sizes([p for p in img_dir.iterdir() if p.suffix.lower() in EXT]))
print("YOLO    :", sizes(yolo))