from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SPLITS_DIR = ROOT / "data" / "splits"
CACHE_DIR = ROOT / "data" / "cache"
OOD_DIR = ROOT / "data" / "ood"
FIELD_DIR = ROOT / "data" / "field"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
CONFIG_JSON = ROOT / "config.json"

CLASSES = ["healthy", "rust", "miner", "cercospora", "phoma"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

SEED = 42
CACHE_SIZE = 256          # tamanho do cache (as fotos são reduzidas uma vez)
IMG_SIZE = 224            # entrada da ResNet50

BATCH_SIZE = 16
NUM_WORKERS = 2
EPOCHS_HEAD = 3           # fase 1: só a camada final
EPOCHS_FINETUNE = 12      # fase 2: layer3 + layer4 + camada final
PATIENCE = 4              # para se o F1 macro da validação não melhorar
LR_HEAD = 1e-3
LR_FINETUNE = 1e-4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]