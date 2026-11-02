import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from config import (SPLITS_DIR, MODELS_DIR, REPORTS_DIR, CLASSES, NUM_CLASSES, SEED,
                    IMG_SIZE, BATCH_SIZE, NUM_WORKERS, EPOCHS_HEAD, EPOCHS_FINETUNE,
                    PATIENCE, LR_HEAD, LR_FINETUNE, IMAGENET_MEAN, IMAGENET_STD)
from dataset import LeafDataset
from model import build_model, set_trainable, set_train_mode


def run_epoch(model, loader, criterion, optimizer=None, phase=None):
    train = optimizer is not None
    if train:
        set_train_mode(model, phase)
    else:
        model.eval()
    total, y_true, y_pred = 0.0, [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            y_true += y.tolist()
            y_pred += out.argmax(1).tolist()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return total / len(y_true), float((y_true == y_pred).mean()), \
        float(f1_score(y_true, y_pred, average="macro"))


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(os.cpu_count())
    MODELS_DIR.mkdir(exist_ok=True)
    (REPORTS_DIR / "metrics").mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "figures").mkdir(parents=True, exist_ok=True)

    train_ds = LeafDataset(SPLITS_DIR / "train.csv", train=True)
    val_ds = LeafDataset(SPLITS_DIR / "val.csv")
    kw = dict(num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, **kw)
    val_dl = DataLoader(val_ds, BATCH_SIZE, shuffle=False, **kw)

    counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES)
    weights = counts.sum() / (NUM_CLASSES * counts)
    print("Treino por classe:", dict(zip(CLASSES, counts.tolist())))
    print("Pesos:", dict(zip(CLASSES, np.round(weights, 2).tolist())))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))

    model = build_model(pretrained=True)
    history, best_f1, bad, epoch_global = [], -1.0, 0, 0

    for phase, n_epochs, lr in [("head", EPOCHS_HEAD, LR_HEAD),
                                ("finetune", EPOCHS_FINETUNE, LR_FINETUNE)]:
        set_trainable(model, phase)
        params = [p for p in model.parameters() if p.requires_grad]
        if phase == "head":
            optimizer = torch.optim.Adam(params, lr=lr)
            scheduler = None
        else:
            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
            bad = 0

        for ep in range(1, n_epochs + 1):
            t0 = time.time()
            epoch_global += 1
            tr_loss, tr_acc, tr_f1 = run_epoch(model, train_dl, criterion, optimizer, phase)
            va_loss, va_acc, va_f1 = run_epoch(model, val_dl, criterion)
            if scheduler:
                scheduler.step()
            history.append(dict(epoch=epoch_global, phase=phase, train_loss=tr_loss,
                                train_acc=tr_acc, train_f1=tr_f1, val_loss=va_loss,
                                val_acc=va_acc, val_f1=va_f1))
            improved = va_f1 > best_f1
            if improved:
                best_f1, bad = va_f1, 0
                torch.save(model.state_dict(), MODELS_DIR / "best.pt")
                (MODELS_DIR / "best_meta.json").write_text(json.dumps({
                    "classes": CLASSES, "img_size": IMG_SIZE, "mean": IMAGENET_MEAN,
                    "std": IMAGENET_STD, "epoch": epoch_global, "val_f1": va_f1}, indent=2))
            else:
                bad += 1
            torch.save(model.state_dict(), MODELS_DIR / "last.pt")
            pd.DataFrame(history).to_csv(REPORTS_DIR / "metrics" / "history.csv", index=False)
            print(f"[{phase} {ep}/{n_epochs}] {time.time() - t0:.0f}s | "
                  f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                  f"val loss {va_loss:.3f} acc {va_acc:.3f} F1 {va_f1:.3f}"
                  f"{'  <- melhor' if improved else ''}", flush=True)
            if phase == "finetune" and bad >= PATIENCE:
                print("Parada antecipada.")
                break

    h = pd.DataFrame(history)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="treino")
    ax[0].plot(h["epoch"], h["val_loss"], label="validação")
    ax[0].set_title("Loss"); ax[0].set_xlabel("época"); ax[0].legend()
    ax[1].plot(h["epoch"], h["train_f1"], label="treino")
    ax[1].plot(h["epoch"], h["val_f1"], label="validação")
    ax[1].set_title("F1 macro"); ax[1].set_xlabel("época"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "curves.png", dpi=150)
    print(f"\nMelhor F1 macro na validação: {best_f1:.3f} (models/best.pt)")


if __name__ == "__main__":
    main()