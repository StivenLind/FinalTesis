# =========================================================
# SISTEMA DE MEDICIÓN DE ÁNGULOS CORPORALES (VERSIÓN CORREGIDA)
# Técnica: Golpe de antebrazos en voleibol
# =========================================================

# Requisitos:
# pip install opencv-python mediapipe pandas numpy

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# ---------------- CONFIGURACIÓN ----------------
IMAGES_PATH = "ImagenesTesis"          # Carpeta real de imágenes
OUTPUT_CSV = "dataset_angulos_voleibol.csv"

# Rangos biomecánicamente correctos
RANGOS = {
    "angulo_brazos": (150, 180),   # Codos extendidos (plataforma)
    "angulo_rodilla": (90, 135),   # Flexión funcional
    "angulo_tronco": (110, 160)    # Inclinación del tronco (ángulo interno)
}

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

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

# ---------------- PROCESAMIENTO ----------------
data = []

for img_name in os.listdir(IMAGES_PATH):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(IMAGES_PATH, img_name)
    image = cv2.imread(img_path)

    if image is None:
        continue

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        continue

    lm = results.pose_landmarks.landmark

    # Puntos clave (lado izquierdo)
    hombro = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
              lm[mp_pose.PoseLandmark.LEFT_SHOULDER].y]

    cadera = [lm[mp_pose.PoseLandmark.LEFT_HIP].x,
              lm[mp_pose.PoseLandmark.LEFT_HIP].y]

    rodilla = [lm[mp_pose.PoseLandmark.LEFT_KNEE].x,
               lm[mp_pose.PoseLandmark.LEFT_KNEE].y]

    tobillo = [lm[mp_pose.PoseLandmark.LEFT_ANKLE].x,
               lm[mp_pose.PoseLandmark.LEFT_ANKLE].y]

    codo = [lm[mp_pose.PoseLandmark.LEFT_ELBOW].x,
            lm[mp_pose.PoseLandmark.LEFT_ELBOW].y]

    muñeca = [lm[mp_pose.PoseLandmark.LEFT_WRIST].x,
              lm[mp_pose.PoseLandmark.LEFT_WRIST].y]

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
        img_name,
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
