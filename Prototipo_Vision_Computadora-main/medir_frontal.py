# =========================================================
# GENERADOR DE DATASET FRONTAL PARA ANTEBRAZOS (VOLEIBOL)
# Usa MediaPipe para extraer features de simetria y alineacion frontal.
# =========================================================

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd

IMAGES_PATH = "Entrenamiento Data Set"
OUTPUT_CSV = "dataset_frontal_voleibol.csv"

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


def distance_xy(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def point_xy(landmarks, idx):
    p = landmarks[idx]
    return [float(p.x), float(p.y)]


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
    original_count = 0
    augmented_count = 0

    os.makedirs(IMAGES_PATH, exist_ok=True)

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
            if not hasattr(results, "pose_landmarks") or not results.pose_landmarks:
                continue

            landmarks = results.pose_landmarks[0]
            features = build_features(landmarks)

            if aug_name == "orig":
                row_name = img_name
                original_count += 1
            else:
                stem, ext = os.path.splitext(img_name)
                row_name = f"{stem}_{aug_name}{ext}"
                augmented_count += 1

            row = {"imagen": row_name}
            row.update(features)
            row["clasificacion"] = auto_label(features)
            rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Dataset frontal generado correctamente: {OUTPUT_CSV}")
    print(f"Muestras originales detectadas: {original_count}")
    print(f"Muestras aumentadas agregadas: {augmented_count}")
    print(f"Total de filas guardadas: {len(df)}")


if __name__ == "__main__":
    main()
