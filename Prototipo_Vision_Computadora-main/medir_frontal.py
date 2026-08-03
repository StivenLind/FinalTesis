# =========================================================
# GENERADOR DE DATASET FRONTAL PARA ANTEBRAZOS (VOLEIBOL)
# Usa MediaPipe para extraer features de simetria y alineacion frontal.
# =========================================================

import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

IMAGES_PATH = "ImagenesTesis"
OUTPUT_CSV = "dataset_frontal_voleibol.csv"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)


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


def main():
    rows = []

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

        landmarks = results.pose_landmarks.landmark
        features = build_features(landmarks)

        row = {"imagen": img_name}
        row.update(features)
        row["clasificacion"] = auto_label(features)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset frontal generado correctamente: {OUTPUT_CSV}")
    print(f"Total de muestras: {len(df)}")


if __name__ == "__main__":
    main()
