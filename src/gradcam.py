import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from config import MODELS_DIR
from dataset import EVAL_TF
from model import build_model


def load_model():
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(MODELS_DIR / "best.pt", map_location="cpu"))
    return m.eval()


def overlay_heat(disp, heat, thr=0.35, alpha=0.6):
    """Pinta só as regiões com calor acima de thr; o resto fica a foto original."""
    base = np.asarray(disp, dtype=np.float32) / 255.0
    colored = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[..., ::-1] / 255.0
    w = (np.clip((heat - thr) / (1 - thr), 0, 1) * alpha)[..., None]
    return Image.fromarray(np.uint8(255 * (base * (1 - w) + colored * w)))


class Explainer:
    def __init__(self, model=None):
        self.model = model if model is not None else load_model()
        self.cam = GradCAMPlusPlus(model=self.model,
                                   target_layers=[self.model.layer3[-1]])

    def __call__(self, img, target=None, width=640):
        """Devolve (probs, classe prevista, overlay, mapa 0-1, foto redimensionada)."""
        img = img.convert("RGB")
        x = EVAL_TF(img).unsqueeze(0)
        with torch.no_grad():
            probs = torch.softmax(self.model(x), 1)[0].numpy()
        pred = int(probs.argmax())
        cls = pred if target is None else int(target)

        heat = self.cam(input_tensor=x, targets=[ClassifierOutputTarget(cls)])[0]
        w, h = img.size
        disp = img.resize((width, round(h * width / w)), Image.LANCZOS)
        heat = np.clip(cv2.resize(heat, disp.size, interpolation=cv2.INTER_CUBIC), 0, 1)
        overlay = overlay_heat(disp, heat)
        return probs, pred, overlay, heat, disp