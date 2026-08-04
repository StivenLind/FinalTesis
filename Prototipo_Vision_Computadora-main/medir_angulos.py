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

# ---------------- CONFIGURACIÓN ----------------
IMAGES_PATH = "Entrenamiento Data Set"  # Carpeta real de imágenes existente en el proyecto
OUTPUT_CSV = "dataset_angulos_voleibol.csv"

# Rangos biomecánicamente correctos
RANGOS = {
    "angulo_brazos": (150, 180),   # Codos extendidos (plataforma)
    "angulo_rodilla": (90, 135),   # Flexión funcional
    "angulo_tronco": (110, 160)    # Inclinación del tronco (ángulo interno)
}

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

POSE_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")
if not os.path.exists(POSE_MODEL_PATH):
    raise FileNotFoundError(
        f"Falta el modelo de MediaPipe pose en: {POSE_MODEL_PATH}. Ejecuta app.py o descarga el archivo."
    )

pose_landmarker = PoseLandmarker.create_from_options(
    PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=POSE_MODEL_PATH),
        running_mode=VisionRunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
)

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


def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def augment_image_variants(image):
    yield "orig", image
    yield "flip", cv2.flip(image, 1)
    yield "rot_pos15", rotate_image(image, 15)
    yield "rot_neg15", rotate_image(image, -15)
    yield "bright_low", adjust_brightness(image, 0.85)
    yield "bright_high", adjust_brightness(image, 1.15)


def save_augmented_image(image, img_name, aug_name, output_dir):
    if aug_name == "orig":
        return None

    stem, ext = os.path.splitext(img_name)
    output_name = f"aug_{stem}_{aug_name}{ext}"
    output_path = os.path.join(output_dir, output_name)
    cv2.imwrite(output_path, image)
    return output_path


# ---------------- PROCESAMIENTO ----------------
os.makedirs(IMAGES_PATH, exist_ok=True)

data = []
original_count = 0
augmented_count = 0

for img_name in os.listdir(IMAGES_PATH):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(IMAGES_PATH, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    for aug_name, aug_image in augment_image_variants(image):
        if aug_name != "orig":
            save_augmented_image(aug_image, img_name, aug_name, IMAGES_PATH)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(aug_image, cv2.COLOR_BGR2RGB),
        )
        results = pose_landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            continue

        lm = results.pose_landmarks[0]

        # Puntos clave (lado izquierdo)
        hombro = [lm[vision.PoseLandmark.LEFT_SHOULDER].x,
                  lm[vision.PoseLandmark.LEFT_SHOULDER].y]

        cadera = [lm[vision.PoseLandmark.LEFT_HIP].x,
                  lm[vision.PoseLandmark.LEFT_HIP].y]

        rodilla = [lm[vision.PoseLandmark.LEFT_KNEE].x,
                   lm[vision.PoseLandmark.LEFT_KNEE].y]

        tobillo = [lm[vision.PoseLandmark.LEFT_ANKLE].x,
                   lm[vision.PoseLandmark.LEFT_ANKLE].y]

        codo = [lm[vision.PoseLandmark.LEFT_ELBOW].x,
                lm[vision.PoseLandmark.LEFT_ELBOW].y]

        muñeca = [lm[vision.PoseLandmark.LEFT_WRIST].x,
                  lm[vision.PoseLandmark.LEFT_WRIST].y]

        # ---------------- ÁNGULOS ----------------
        angulo_brazos = calcular_angulo(hombro, codo, muñeca)
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

        if aug_name == "orig":
            row_name = img_name
            original_count += 1
        else:
            stem, ext = os.path.splitext(img_name)
            row_name = f"{stem}_{aug_name}{ext}"
            augmented_count += 1

        data.append([
            row_name,
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
print(f"Muestras originales detectadas: {original_count}")
print(f"Muestras aumentadas agregadas: {augmented_count}")
print(f"Total de filas guardadas: {len(df)}")
