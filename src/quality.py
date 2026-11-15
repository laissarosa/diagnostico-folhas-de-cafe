import cv2
import numpy as np
from PIL import Image


def sharpness(img: Image.Image, width: int = 512) -> float:
    """Variância do Laplaciano em tons de cinza, com largura padronizada.
    Quanto menor, mais borrada a foto."""
    g = img.convert("L")
    w, h = g.size
    g = g.resize((width, max(1, round(h * width / w))), Image.BILINEAR)
    return float(cv2.Laplacian(np.asarray(g), cv2.CV_64F).var())
