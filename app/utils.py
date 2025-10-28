import cv2
import numpy as np

def measure_blur(gray_image: np.ndarray) -> float:
    """Variação do Laplaciano - medida simples de nitidez."""
    return float(cv2.Laplacian(gray_image, cv2.CV_64F).var())

def mean_brightness(gray_image: np.ndarray) -> float:
    """Média dos pixels (0-255)."""
    return float(gray_image.mean())

def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / float(max(h, w))
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(img, (new_w, new_h))