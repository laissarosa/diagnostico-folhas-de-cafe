import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import SPLITS_DIR, MODELS_DIR, REPORTS_DIR, CLASSES
from dataset import LeafDataset
from model import build_model


@torch.no_grad()
def predict(model, ds):
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    return np.concatenate([torch.softmax(model(x), 1).numpy() for x, _ in dl])


def main():
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(MODELS_DIR / "best.pt", map_location="cpu"))
    model.eval()
    (REPORTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    for split in ["val", "test"]:
        ds = LeafDataset(SPLITS_DIR / f"{split}.csv")
        probs = predict(model, ds)
        y, pred = np.array(ds.labels), probs.argmax(1)

        out = ds.df[["path", "label", "label_idx"]].copy()
        out["pred"] = pred
        out["conf"] = probs.max(1)
        for i, c in enumerate(CLASSES):
            out[f"p_{c}"] = probs[:, i]
        out.to_csv(REPORTS_DIR / "metrics" / f"{split}_probs.csv", index=False)

        print(f"\n===== {split.upper()} =====")
        print(classification_report(y, pred, target_names=CLASSES, digits=3, zero_division=0))

        cm = confusion_matrix(y, pred, labels=range(len(CLASSES)))
        cmn = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=30)
        ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
        ax.set_xlabel("previsto"); ax.set_ylabel("real"); ax.set_title(f"Matriz de confusão ({split})")
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                ax.text(j, i, f"{cm[i, j]}\n{cmn[i, j]:.0%}", ha="center", va="center",
                        color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
        fig.tight_layout()
        fig.savefig(REPORTS_DIR / "figures" / f"confusion_{split}.png", dpi=150)


if __name__ == "__main__":
    main()