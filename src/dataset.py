import hashlib

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import CACHE_DIR, CACHE_SIZE, IMG_SIZE, IMAGENET_MEAN, IMAGENET_STD

NORM = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)

TRAIN_TF = T.Compose([
    T.RandomResizedCrop(IMG_SIZE, scale=(0.6, 1.0), ratio=(0.9, 1.1)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomApply([T.RandomRotation(25)], p=0.5),
    T.ColorJitter(0.3, 0.3, 0.2, 0.02),
    T.ToTensor(),
    NORM,
])

# Também será usado na API: serve para foto original ou já em cache
EVAL_TF = T.Compose([
    T.Resize((CACHE_SIZE, CACHE_SIZE)),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    NORM,
])


def cache_name(path: str) -> str:
    return hashlib.md5(path.encode()).hexdigest()[:12] + ".jpg"


class LeafDataset(Dataset):
    def __init__(self, csv_path, train=False):
        self.df = pd.read_csv(csv_path)
        self.files = [CACHE_DIR / cache_name(p) for p in self.df["path"]]
        self.labels = self.df["label_idx"].tolist()
        self.tf = TRAIN_TF if train else EVAL_TF

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        return self.tf(img), self.labels[i]