import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import ROOT, REPORTS_DIR, CONFIG_JSON
from quality import sharpness

TARGET_ACC = 0.95        # acurácia mínima desejada nos casos aceitos (validação)
SHARP_PERCENTILE = 1     # percentil da nitidez (validação) usado como mínimo

val = pd.read_csv(REPORTS_DIR / "metrics" / "val_probs.csv")
test = pd.read_csv(REPORTS_DIR / "metrics" / "test_probs.csv")


def curve(df, thresholds):
    ok = (df["label_idx"] == df["pred"]).values
    conf = df["conf"].values
    rows = []
    for t in thresholds:
        acc_mask = conf >= t
        rows.append({
            "threshold": round(float(t), 3),
            "coverage": acc_mask.mean(),
            "acc_accepted": ok[acc_mask].mean() if acc_mask.any() else np.nan,
            "errors_refused": int((~ok & ~acc_mask).sum()),
            "errors_total": int((~ok).sum()),
        })
    return pd.DataFrame(rows)


grid = np.arange(0.30, 0.9951, 0.005)
cv, ct = curve(val, grid), curve(test, grid)

good = cv[cv["acc_accepted"] >= TARGET_ACC]
if len(good):
    threshold = float(good.iloc[0]["threshold"])
else:
    threshold = float(cv.loc[cv["acc_accepted"].idxmax(), "threshold"])
    print(f"Atenção: nenhum limiar atinge {TARGET_ACC:.0%} na validação; usando o de maior acurácia.")

# Nitidez nas imagens de validação
sharps = []
for p in val["path"]:
    with Image.open(ROOT / p) as im:
        im.draft("RGB", (1024, 512))
        sharps.append(sharpness(im.convert("RGB")))
sharps = np.array(sharps)
min_sharp = float(np.percentile(sharps, SHARP_PERCENTILE))

CONFIG_JSON.write_text(json.dumps({
    "threshold": threshold, "min_sharpness": min_sharp,
    "target_acc": TARGET_ACC, "sharpness_percentile": SHARP_PERCENTILE,
}, indent=2))
print(f"Limiar de confiança: {threshold}  |  nitidez mínima: {min_sharp:.1f} "
      f"(mediana na validação: {np.median(sharps):.1f})  -> config.json")

# Relatório com o limiar fixado na validação
for name, df in [("VALIDAÇÃO", val), ("TESTE", test)]:
    ok = (df["label_idx"] == df["pred"]).values
    acc_mask = df["conf"].values >= threshold
    n_err, n_ref = int((~ok).sum()), int((~acc_mask).sum())
    print(f"\n===== {name} (limiar {threshold}) =====")
    print(f"Taxa de recusa:                {(~acc_mask).mean():.1%} ({n_ref} de {len(df)})")
    print(f"Acurácia nos aceitos:          {ok[acc_mask].mean():.1%}")
    if n_ref:
        print(f"Acurácia nos recusados:        {ok[~acc_mask].mean():.1%}")
    print(f"Erros do modelo recusados:     {int((~ok & ~acc_mask).sum())} de {n_err}")
    print(f"Acurácia sem recusa (todos):   {ok.mean():.1%}")

(REPORTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
cv.to_csv(REPORTS_DIR / "metrics" / "coverage_val.csv", index=False)
ct.to_csv(REPORTS_DIR / "metrics" / "coverage_test.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4.5))
ax.plot(cv["coverage"], cv["acc_accepted"], label="validação")
ax.plot(ct["coverage"], ct["acc_accepted"], label="teste")
t_row = ct.iloc[(ct["threshold"] - threshold).abs().argmin()]
ax.scatter([t_row["coverage"]], [t_row["acc_accepted"]], color="red", zorder=5,
           label=f"limiar {threshold}")
ax.set_xlabel("cobertura (fração aceita)")
ax.set_ylabel("acurácia nos casos aceitos")
ax.set_title("Cobertura x acurácia")
ax.grid(alpha=.3)
ax.legend()
fig.tight_layout()
(REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
fig.savefig(REPORTS_DIR / "figures" / "coverage_accuracy.png", dpi=150)
