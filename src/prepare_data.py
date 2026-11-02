# src/prepare_data.py  (Mendeley + YOLO, sem fotos duplicadas)

'''
Se toda saudável vier de uma fonte e toda doente da outra, a origem da imagem prevê a
classe, e o modelo poderia usar diferenças de exportação (compressão, orientação) como
atalho. Em vez disso, o script abaixo usa o Mendeley para todas as classes e só 
completa com imagens do YOLO que não existem no Mendeley. Ele identifica a mesma foto nas
 duas fontes por uma impressão digital da imagem (dHash), então não entram cópias duplicadas,
que causariam vazamento entre treino e teste. Assim, as doentes vêm das duas fontes e a saudável
 só do Mendeley, e a origem deixa de prever a classe na maioria dos casos.
'''
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from config import ROOT, SEED, SPLITS_DIR

MEND = ROOT / "data" / "raw" / "bracol_manual"
YOLO = ROOT / "data" / "raw" / "BRACOL-ORIGINAL-ANNOTATIONS" / "BRACOL-ORIGINAL-DETECT"

STRESS_TO_CLASS = {0: "healthy", 1: "miner", 2: "rust", 3: "phoma", 4: "cercospora"}
YOLO_NAMES = ["cercospora", "miner", "phoma", "rust"]          # ordem do data.yaml
CLASSES = ["healthy", "rust", "miner", "cercospora", "phoma"]  # igual ao config.py
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IMG_EXT = {".jpg", ".jpeg", ".png"}

AMBIGUOUS_RATIO = 0.40     # YOLO: 2ª doença > 40% da área da 1ª = ambígua (removida)
DUP_THRESHOLD = 24         # distância (de 512 bits) abaixo da qual é a mesma foto


def dhash(path):
    with Image.open(path) as im:
        im.draft("L", (264, 132))
        g = im.convert("L").resize((33, 16), Image.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()            # 512 bits


def with_hashes(df):
    hs, keep = [], []
    for p in df["abs"]:
        try:
            hs.append(dhash(p))
            keep.append(True)
        except Exception:
            keep.append(False)
    bad = df.loc[[not k for k in keep], "abs"].map(lambda p: p.name).tolist()
    return df[keep].reset_index(drop=True), np.stack(hs).astype(np.float32), bad


def load_mendeley():
    csv = next(MEND.rglob("dataset.csv"))
    img_dir = max((d for d in MEND.rglob("images") if d.is_dir()),
                  key=lambda d: sum(1 for _ in d.iterdir()))
    df = pd.read_csv(csv).dropna(subset=["id", "predominant_stress"]).copy()
    df["id"] = df["id"].astype(int)
    df["predominant_stress"] = df["predominant_stress"].astype(int)
    n0 = len(df)
    df = df[df["predominant_stress"].isin(STRESS_TO_CLASS)].copy()
    print(f"Mendeley: {n0} linhas no CSV, {n0 - len(df)} sem rótulo 0-4 (descartadas)")
    files = {p.stem: p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT}
    df["abs"] = df["id"].astype(str).map(files)
    print(f"Mendeley: {int(df['abs'].isna().sum())} linhas sem imagem no disco")
    df = df.dropna(subset=["abs"]).copy()
    df["label"] = df["predominant_stress"].map(STRESS_TO_CLASS)
    df["group"] = "m" + df["id"].astype(str)
    df["source"] = "mendeley"
    return df[["abs", "label", "group", "source"]].reset_index(drop=True)


def parse_label(txt):
    areas, n = defaultdict(float), 0
    if not txt.exists():
        return areas, n
    for line in txt.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, v = int(float(p[0])), [float(x) for x in p[1:]]
        if c >= len(YOLO_NAMES):
            continue
        if len(p) == 5:
            area = v[2] * v[3]
        else:
            xs, ys = v[0::2], v[1::2]
            m = len(xs)
            area = 0.5 * abs(sum(xs[i] * ys[(i + 1) % m] - xs[(i + 1) % m] * ys[i]
                                 for i in range(m)))
        areas[YOLO_NAMES[c]] += area
        n += 1
    return areas, n


def load_yolo():
    rows, amb = [], 0
    for split in ["train", "valid", "test"]:
        img_dir, lab_dir = YOLO / split / "images", YOLO / split / "labels"
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMG_EXT:
                continue
            areas, n = parse_label(lab_dir / (img.stem + ".txt"))
            if n == 0:
                continue
            ranked = sorted(areas.items(), key=lambda kv: kv[1], reverse=True)
            ratio = ranked[1][1] / ranked[0][1] if len(ranked) > 1 and ranked[0][1] > 0 else 0.0
            if ratio > AMBIGUOUS_RATIO:
                amb += 1
                continue
            rows.append({"abs": img, "label": ranked[0][0],
                         "group": "y" + img.stem.split(".rf.")[0], "source": "yolo"})
    print(f"YOLO: {len(rows)} imagens (ambíguas removidas: {amb})")
    return pd.DataFrame(rows)


def main():
    m, y = load_mendeley(), load_yolo()

    print("\nCalculando impressões digitais (1 a 3 min)...")
    m, Hm, bad_m = with_hashes(m)
    y, Hy, bad_y = with_hashes(y)
    print("Corrompidas no Mendeley:", bad_m)
    print("Corrompidas no YOLO:", bad_y)

    # Mesma foto nas duas fontes?
    dist = (Hy.sum(1)[:, None] + Hm.sum(1)[None, :]) - 2 * Hy @ Hm.T
    nearest, arg = dist.min(1), dist.argmin(1)
    print("\nDistância da foto do YOLO mais parecida no Mendeley (nº de fotos YOLO com distância <= t):")
    for t in (0, 8, 16, 24, 48, 96, 160):
        print(f"  <= {t:3d}: {int((nearest <= t).sum())}")
    dup = nearest <= DUP_THRESHOLD
    print(f"\nFotos do YOLO que já existem no Mendeley (descartadas): {int(dup.sum())}")
    if dup.any():
        agree = y.loc[dup, "label"].values == m["label"].values[arg[dup]]
        print(f"Rótulos iguais entre as fontes nessas fotos: {int(agree.sum())}/{int(dup.sum())}")
        print(pd.crosstab(y.loc[dup, "label"].values, m["label"].values[arg[dup]],
                          rownames=["YOLO"], colnames=["Mendeley"]).to_string())
    y = y[~dup]

    df = pd.concat([m, y], ignore_index=True)
    df["label_idx"] = df["label"].map(CLASS_TO_IDX)
    df["path"] = df["abs"].map(lambda p: p.relative_to(ROOT).as_posix())

    print(f"\nTotal: {len(df)} imagens")
    print(pd.crosstab(df["label"], df["source"], margins=True).to_string())

    sgkf = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=SEED)
    df["fold"] = -1
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], df["group"])):
        df.loc[df.index[idx], "fold"] = k
    df["split"] = df["fold"].map(lambda k: "test" if k < 3 else "val" if k < 6 else "train")

    cols = ["path", "label", "label_idx", "source", "group", "split"]
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(SPLITS_DIR / "all.csv", index=False)
    for s in ["train", "val", "test"]:
        df[df["split"] == s][cols].to_csv(SPLITS_DIR / f"{s}.csv", index=False)

    print("\nSplit final:")
    print(pd.crosstab(df["split"], df["label"], margins=True).to_string())


if __name__ == "__main__":
    main()