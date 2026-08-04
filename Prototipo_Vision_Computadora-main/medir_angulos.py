# =========================================================
# SISTEMA DE MEDICIÓN DE ÁNGULOS CORPORALES (VERSIÓN CORREGIDA)
# Técnica: Golpe de antebrazos en voleibol
# =========================================================

# Requisitos:
# pip install opencv-python mediapipe pandas numpy

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import os
from pathlib import Path

# ---------------- CONFIGURACIÓN ----------------
IMAGES_PATH = "Entrenamiento Data Set"  # Carpeta del dataset (originales + aumentadas)
OUTPUT_CSV = "dataset_angulos_voleibol.csv"
EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

# Rangos biomecánicamente correctos
RANGOS = {
    "angulo_brazos": (150, 180),   # Codos extendidos (plataforma)
    "angulo_rodilla": (90, 135),   # Flexión funcional
    "angulo_tronco": (110, 160)    # Inclinación del tronco (ángulo interno)
}

BASE_DIR = Path(__file__).resolve().parent
POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23
LEFT_KNEE = 25
LEFT_ANKLE = 27

# ---------------- FUNCIONES ----------------

def calcular_angulo(a, b, c):
    """Calcula el ángulo ABC en grados"""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
    return angle


def evaluar_rango(valor, rango):
    return rango[0] <= valor <= rango[1]


def es_imagen(path: Path):
    return path.suffix.lower() in EXTENSIONES_IMAGEN


def leer_imagen(path: Path):
    # Evita fallos de cv2.imread con rutas con espacios o caracteres especiales.
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

# ---------------- PROCESAMIENTO ----------------
data = []

dataset_dir = Path(IMAGES_PATH)
if not dataset_dir.exists() or not dataset_dir.is_dir():
    raise FileNotFoundError(f"No se encontro la carpeta del dataset: {dataset_dir}")

imagenes = [p for p in dataset_dir.rglob("*") if p.is_file() and es_imagen(p)]
total_imagenes = len(imagenes)
fallos_lectura = 0
sin_pose = 0
con_pose = 0

if not POSE_MODEL_PATH.exists():
    raise FileNotFoundError(f"No se encontro el modelo de pose: {POSE_MODEL_PATH}")

base_options = python.BaseOptions(model_asset_path=str(POSE_MODEL_PATH))
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_poses=1,
)

with vision.PoseLandmarker.create_from_options(options) as pose:
    for img_path in imagenes:
        image = leer_imagen(img_path)

        if image is None:
            fallos_lectura += 1
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = pose.detect(mp_image)

        if not results.pose_landmarks:
            sin_pose += 1
            continue

        con_pose += 1
        lm = results.pose_landmarks[0]

        # Puntos clave (lado izquierdo)
        hombro = [lm[LEFT_SHOULDER].x, lm[LEFT_SHOULDER].y]
        cadera = [lm[LEFT_HIP].x, lm[LEFT_HIP].y]
        rodilla = [lm[LEFT_KNEE].x, lm[LEFT_KNEE].y]
        tobillo = [lm[LEFT_ANKLE].x, lm[LEFT_ANKLE].y]
        codo = [lm[LEFT_ELBOW].x, lm[LEFT_ELBOW].y]
        muñeca = [lm[LEFT_WRIST].x, lm[LEFT_WRIST].y]

        # ---------------- ÁNGULOS ----------------
        angulo_brazos = calcular_angulo(hombro, codo, muñeca)   # Extensión del codo
        angulo_rodilla = calcular_angulo(cadera, rodilla, tobillo)
        angulo_tronco = calcular_angulo(hombro, cadera, rodilla)

        # ---------------- EVALUACIÓN FLEXIBLE ----------------
        criterios = {
            "brazos_ok": evaluar_rango(angulo_brazos, RANGOS["angulo_brazos"]),
            "rodilla_ok": evaluar_rango(angulo_rodilla, RANGOS["angulo_rodilla"]),
            "tronco_ok": evaluar_rango(angulo_tronco, RANGOS["angulo_tronco"])
        }

        criterios_cumplidos = sum(criterios.values())
        clasificacion = "CORRECTO" if criterios_cumplidos >= 2 else "INCORRECTO"

        data.append([
            str(img_path.relative_to(dataset_dir)),
            round(angulo_brazos, 2),
            round(angulo_rodilla, 2),
            round(angulo_tronco, 2),
            criterios_cumplidos,
            clasificacion
        ])

# ---------------- DATASET ----------------
df = pd.DataFrame(data, columns=[
    "imagen",
    "angulo_brazos",
    "angulo_rodilla",
    "angulo_tronco",
    "criterios_cumplidos",
    "clasificacion"
])

df.to_csv(OUTPUT_CSV, index=False)
print("Dataset generado correctamente:", OUTPUT_CSV)
print(f"Imagenes encontradas: {total_imagenes}")
print(f"Con pose detectada: {con_pose}")
print(f"Sin pose detectada: {sin_pose}")
print(f"Fallos de lectura: {fallos_lectura}")
