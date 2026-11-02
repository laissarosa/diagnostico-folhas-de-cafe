import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from config import ROOT, REPORTS_DIR, CLASSES, SEED
from gradcam import Explainer

N_OK, N_ERR, N_REVIEW_OK = 3, 3, 6
FIG_DIR = REPORTS_DIR / "figures" / "gradcam"
REV_DIR = REPORTS_DIR / "gradcam_review"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REV_DIR.mkdir(parents=True, exist_ok=True)

ex = Explainer()
df = pd.read_csv(REPORTS_DIR / "metrics" / "test_probs.csv")
df["ok"] = df["label_idx"] == df["pred"]


def panel(rows, title, fname):
    fig, axes = plt.subplots(len(rows), 2, figsize=(9, 2.6 * len(rows)), squeeze=False)
    for ax_row, r in zip(axes, rows.itertuples()):
        probs, pred, overlay, _, disp = ex(Image.open(ROOT / r.path))
        ax_row[0].imshow(disp)
        ax_row[1].imshow(overlay)
        ax_row[0].set_title(f"real: {CLASSES[r.label_idx]}", fontsize=9)
        ax_row[1].set_title(f"previsto: {CLASSES[pred]} ({probs[pred]:.0%})", fontsize=9)
        for a in ax_row:
            a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(FIG_DIR / fname, dpi=130)
    plt.close(fig)


# 1) Figuras por classe para o pôster
for i, c in enumerate(CLASSES):
    ok = df[(df["label_idx"] == i) & df["ok"]]
    err = df[(df["label_idx"] == i) & ~df["ok"]]
    if len(ok):
        panel(ok.sample(min(N_OK, len(ok)), random_state=SEED), f"{c}: acertos", f"{c}_acertos.png")
    if len(err):
        panel(err.sample(min(N_ERR, len(err)), random_state=SEED), f"{c}: erros", f"{c}_erros.png")

# 2) Folha de revisão manual
review = pd.concat([
    df[~df["ok"]],
    df[df["ok"]].groupby("label_idx", group_keys=False)
      .apply(lambda g: g.sample(min(N_REVIEW_OK, len(g)), random_state=SEED)),
]).reset_index(drop=True)

rows = []
for n, r in enumerate(review.itertuples()):
    probs, pred, overlay, _, disp = ex(Image.open(ROOT / r.path))
    sheet = Image.new("RGB", (disp.width * 2, disp.height))
    sheet.paste(disp, (0, 0))
    sheet.paste(overlay, (disp.width, 0))
    name = f"{n:02d}_real-{CLASSES[r.label_idx]}_prev-{CLASSES[pred]}.jpg"
    sheet.save(REV_DIR / name, quality=90)
    rows.append({"arquivo": name, "path": r.path, "real": CLASSES[r.label_idx],
                 "previsto": CLASSES[pred], "correto": bool(r.ok), "olhou_lesao": ""})

pd.DataFrame(rows).to_csv(REV_DIR / "gradcam_review.csv", index=False, encoding="utf-8-sig")
print(f"Figuras em {FIG_DIR}")
print(f"{len(rows)} imagens para revisão em {REV_DIR}")