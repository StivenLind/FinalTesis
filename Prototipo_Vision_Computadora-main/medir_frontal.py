# =========================================================
# GENERADOR DE DATASET FRONTAL PARA ANTEBRAZOS (VOLEIBOL)
# Usa MediaPipe para extraer features de simetria y alineacion frontal.
# =========================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

IMAGES_PATH = "Entrenamiento Data Set"
OUTPUT_CSV = "dataset_frontal_voleibol.csv"
EXTENSIONES_IMAGEN = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}

BASE_DIR = Path(__file__).resolve().parent
POSE_MODEL_PATH = BASE_DIR / "pose_landmarker_lite.task"


def distance_xy(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def point_xy(landmarks, idx):
    p = landmarks[idx]
    return [float(p.x), float(p.y)]


def build_features(landmarks):
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    _ = point_xy(landmarks, NOSE)
    hombro_izq = point_xy(landmarks, LEFT_SHOULDER)
    hombro_der = point_xy(landmarks, RIGHT_SHOULDER)
    mano_izq = point_xy(landmarks, LEFT_WRIST)
    mano_der = point_xy(landmarks, RIGHT_WRIST)
    cadera_izq = point_xy(landmarks, LEFT_HIP)
    cadera_der = point_xy(landmarks, RIGHT_HIP)
    rodilla_izq = point_xy(landmarks, LEFT_KNEE)
    rodilla_der = point_xy(landmarks, RIGHT_KNEE)
    tobillo_izq = point_xy(landmarks, LEFT_ANKLE)
    tobillo_der = point_xy(landmarks, RIGHT_ANKLE)

    tronco = [
        (hombro_izq[0] + hombro_der[0]) / 2.0,
        (hombro_izq[1] + hombro_der[1]) / 2.0,
    ]
    cadera = [
        (cadera_izq[0] + cadera_der[0]) / 2.0,
        (cadera_izq[1] + cadera_der[1]) / 2.0,
    ]

    ancho_hombros = distance_xy(hombro_izq, hombro_der)
    scale = max(ancho_hombros, 1e-6)

    return {
        "dist_manos": distance_xy(mano_izq, mano_der) / scale,
        "ancho_caderas": distance_xy(cadera_izq, cadera_der) / scale,
        "ancho_rodillas": distance_xy(rodilla_izq, rodilla_der) / scale,
        "ancho_tobillos": distance_xy(tobillo_izq, tobillo_der) / scale,
        "alineacion_tronco_x": abs(tronco[0] - cadera[0]) / scale,
        "simetria_rodillas_y": abs(rodilla_izq[1] - rodilla_der[1]) / scale,
        "simetria_manos_y": abs(mano_izq[1] - mano_der[1]) / scale,
    }


def auto_label(features):
    score = 0
    score += 1 if features["alineacion_tronco_x"] <= 0.12 else 0
    score += 1 if features["simetria_rodillas_y"] <= 0.08 else 0
    score += 1 if features["simetria_manos_y"] <= 0.08 else 0
    score += 1 if 0.5 <= features["ancho_tobillos"] <= 1.5 else 0
    return "CORRECTO" if score >= 3 else "INCORRECTO"


def es_imagen(path: Path):
    return path.suffix.lower() in EXTENSIONES_IMAGEN


def leer_imagen(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def main():
    rows = []
    dataset_dir = Path(IMAGES_PATH)
    if not dataset_dir.exists() or not dataset_dir.is_dir():
        raise FileNotFoundError(f"No se encontro la carpeta del dataset: {dataset_dir}")

    if not POSE_MODEL_PATH.exists():
        raise FileNotFoundError(f"No se encontro el modelo de pose: {POSE_MODEL_PATH}")

    imagenes = [p for p in dataset_dir.rglob("*") if p.is_file() and es_imagen(p)]
    total_imagenes = len(imagenes)
    con_pose = 0
    sin_pose = 0
    fallos_lectura = 0

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
            landmarks = results.pose_landmarks[0]
            features = build_features(landmarks)

            row = {"imagen": str(img_path.relative_to(dataset_dir))}
            row.update(features)
            row["clasificacion"] = auto_label(features)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset frontal generado correctamente: {OUTPUT_CSV}")
    print(f"Total de muestras: {len(df)}")
    print(f"Imagenes encontradas: {total_imagenes}")
    print(f"Con pose detectada: {con_pose}")
    print(f"Sin pose detectada: {sin_pose}")
    print(f"Fallos de lectura: {fallos_lectura}")


if __name__ == "__main__":
    main()
