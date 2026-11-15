import base64
import io
import json
import sys
import threading
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from config import CLASSES, CONFIG_JSON  # noqa: E402
from gradcam import Explainer  # noqa: E402
from quality import sharpness  # noqa: E402

QUEUE = ROOT / "server" / "queue"
QUEUE.mkdir(parents=True, exist_ok=True)

cfg = json.loads(CONFIG_JSON.read_text())
THRESHOLD, MIN_SHARP = cfg["threshold"], cfg["min_sharpness"]

NAMES_PT = {
    "healthy": "Saudável",
    "rust": "Ferrugem",
    "miner": "Bicho-mineiro",
    "cercospora": "Cercosporiose",
    "phoma": "Mancha de phoma",
}

explainer = Explainer()          # carrega o modelo uma única vez
lock = threading.Lock()          # Grad-CAM usa hooks: uma inferência por vez

app = FastAPI(title="Diagnóstico de doenças do café")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


def read_image(data: bytes) -> Image.Image:
    try:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img).convert("RGB")   # orientação do celular
    except Exception:
        raise HTTPException(status_code=400, detail="O arquivo não é uma imagem válida.")
    img.thumbnail((2048, 2048))
    return img


@app.get("/health")
def health():
    return {"status": "ok", "threshold": THRESHOLD, "min_sharpness": MIN_SHARP}


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    img = read_image(file.file.read())
    sharp = sharpness(img)
    with lock:
        probs, pred, overlay, _, _ = explainer(img)

    conf = float(probs[pred])
    reason = None
    if sharp < MIN_SHARP:
        reason = "imagem_borrada"
    elif conf < THRESHOLD:
        reason = "baixa_confianca"

    buf = io.BytesIO()
    overlay.save(buf, "JPEG", quality=85)
    return {
        "class": CLASSES[pred],
        "class_pt": NAMES_PT[CLASSES[pred]],
        "confidence": conf,
        "refused": reason is not None,
        "reason": reason,
        "sharpness": sharp,
        "threshold": THRESHOLD,
        "probs": {NAMES_PT[c]: float(p) for c, p in zip(CLASSES, probs)},
        "gradcam": base64.b64encode(buf.getvalue()).decode(),
    }


@app.post("/escalate")
def escalate(file: UploadFile = File(...), predicted: str = Form(""),
             confidence: float = Form(0.0), reason: str = Form("")):
    """Simula o encaminhamento ao agrônomo: guarda a foto e o diagnóstico na fila."""
    img = read_image(file.file.read())
    name = time.strftime("%Y%m%d-%H%M%S") + f"-{int(time.time() * 1000) % 1000:03d}"
    img.save(QUEUE / f"{name}.jpg", quality=92)
    (QUEUE / f"{name}.json").write_text(json.dumps({
        "id": name, "predicted": predicted, "confidence": confidence,
        "reason": reason, "status": "aguardando agrônomo",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"status": "enviado", "id": name}
