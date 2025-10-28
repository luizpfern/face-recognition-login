import cv2
import numpy as np
import logging
from typing import Tuple, Optional, List

from app.utils import (
    measure_blur,
    mean_brightness,
    resize_max_side,
)
# face_recognition é usado apenas para extrair embeddings
# Import lazy inside function to facilitar testes sem dlib instalado

# Config simples (ajuste conforme necessário)
BRIGHTNESS_MIN = 50
BRIGHTNESS_MAX = 230
BLUR_THRESHOLD = 80.0
MIN_FACE_AREA_RATIO = 0.05
TARGET_FACE_SIDE = 150
FACE_DISTANCE_THRESHOLD = 0.6
MIN_VOTES_FOR_AUTH = 2  # 2 de 3 por padrão


def decode_image_bytes(data: bytes) -> Optional[np.ndarray]:
    """Decodifica bytes para imagem BGR (OpenCV)."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def detect_and_validate(img_bgr: np.ndarray) -> Tuple[bool, Optional[str]]:
    """
    Detecta (placeholder) e faz checagens simples: brilho, blur, face size.
    Aqui usamos um detector simples por Haar cascade se disponível, mas
    para reduzir dependências, deixamos um fallback.
    """
    if img_bgr is None:
        return False, "invalid_image"

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if mean_brightness(gray) < BRIGHTNESS_MIN:
        return False, "face_dark"
    if mean_brightness(gray) > BRIGHTNESS_MAX:
        return False, "face_overexposed"
    if measure_blur(gray) < BLUR_THRESHOLD:
        return False, "face_blurry"

    # Detector simples: tentar Haar cascade incluída no OpenCV (se não existir, assume face whole)
    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(30, 30))
    except Exception:
        faces = []

    if len(faces) == 0:
        # fallback: assume full image is face (não ideal, mas mantém fluxo simples)
        h, w = img_bgr.shape[:2]
        faces = np.array([[0, 0, w, h]])

    if len(faces) == 0:
        return False, "no_face_detected"
    if len(faces) > 1:
        return False, "multiple_faces"

    x, y, w, h = faces[0]
    area_ratio = (w * h) / (img_bgr.shape[0] * img_bgr.shape[1])
    if area_ratio < MIN_FACE_AREA_RATIO:
        return False, "face_too_small"

    return True, None


def align_face(img_bgr: np.ndarray) -> np.ndarray:
    """
    Simplified align: crop using center bbox heuristics.
    For full implementation, use landmarks to rotate and tightly crop.
    """
    h, w = img_bgr.shape[:2]
    # crop center square around image center as naive face crop
    side = min(h, w)
    cx, cy = w // 2, h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    crop = img_bgr[y1 : y1 + side, x1 : x1 + side]
    return crop


def preprocess_face(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Prepara imagem para embedding: BGR->RGB, resize.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    resized = resize_max_side(rgb, TARGET_FACE_SIDE)
    return resized


def get_embedding(rgb_face: np.ndarray) -> Optional[np.ndarray]:
    """
    Extrai embedding usando face_recognition. Retorna 1-d numpy array ou None.
    """
    try:

        import face_recognition
        print("importei essa bosta")
    except Exception as error:
        # Logar o erro de import de forma segura (evita concatenar objeto Exception)
        logging.warning("face_recognition import error: %s", error)
        # face_recognition não instalado / dlib ausente
        return None
    print("passou")
    encs = face_recognition.face_encodings(rgb_face)
    if not encs:
        return None
    return encs[0]


def compare_embeddings(known_vectors: List[List[float]], candidates: List[np.ndarray]) -> dict:
    """
    Compara candidatos com known_vectors (lista de listas floats).
    Retorna dict com authenticated, votes, avg_distance, distances.
    """
    try:
        import face_recognition
    except Exception:
        return {"authenticated": False, "reason": "face_recognition_not_available"}

    known = [np.array(v) for v in known_vectors]

    votes = 0
    distances = []
    for cand in candidates:
        dists = face_recognition.face_distance(known, cand)
        min_dist = float(np.min(dists))
        distances.append(min_dist)
        if min_dist <= FACE_DISTANCE_THRESHOLD:
            votes += 1

    avg = float(sum(distances) / len(distances)) if distances else None
    authenticated = votes >= MIN_VOTES_FOR_AUTH
    return {"authenticated": authenticated, "votes": votes, "avg_distance": avg, "distances": distances}