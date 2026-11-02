from PIL import Image
from tqdm import tqdm
import pandas as pd

from config import ROOT, SPLITS_DIR, CACHE_DIR, CACHE_SIZE
from dataset import cache_name

CACHE_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(SPLITS_DIR / "all.csv")

for p in tqdm(df["path"], desc="cache"):
    dest = CACHE_DIR / cache_name(p)
    if dest.exists():
        continue
    with Image.open(ROOT / p) as im:
        im.draft("RGB", (CACHE_SIZE * 2, CACHE_SIZE))
        im = im.convert("RGB").resize((CACHE_SIZE, CACHE_SIZE), Image.LANCZOS)
        im.save(dest, quality=95)

print("Cache pronto em", CACHE_DIR, "-", len(list(CACHE_DIR.glob('*.jpg'))), "imagens")