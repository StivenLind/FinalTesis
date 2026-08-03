# =========================================================
# RETROALIMENTACIÓN EN TIEMPO REAL – GOLPE DE ANTEBRAZOS
# Cámara web + MediaPipe + Modelo IA entrenado
# =========================================================

# Requisitos:
# pip install opencv-python mediapipe numpy pandas scikit-learn joblib

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import joblib
import os

# ---------------- CARGA DEL MODELO ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELO_PATH = os.path.join(BASE_DIR, "modelo_tecnica_antebrazos.pkl")

if not os.path.exists(MODELO_PATH):
    raise FileNotFoundError(
        f"No se encontro el modelo en: {MODELO_PATH}. Ejecuta primero modelo_IA.py"
    )

modelo = joblib.load(MODELO_PATH)

# ---------------- MEDIAPIPE ----------------
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.6,
                    min_tracking_confidence=0.6)

# ---------------- FUNCIONES ----------------
def calcular_angulo(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / denom
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


def evaluar_tecnica(ang_brazos, ang_rodilla, ang_tronco):
    entrada = pd.DataFrame([{
        "angulo_brazos": ang_brazos,
        "angulo_rodilla": ang_rodilla,
        "angulo_tronco": ang_tronco
    }])
    pred = modelo.predict(entrada)[0]
    return "CORRECTO" if pred == 1 else "INCORRECTO"


def seleccionar_lado_mas_visible(lm):
    lados = {
        "LEFT": {
            "hombro": lm[mp_pose.PoseLandmark.LEFT_SHOULDER],
            "codo": lm[mp_pose.PoseLandmark.LEFT_ELBOW],
            "muneca": lm[mp_pose.PoseLandmark.LEFT_WRIST],
            "cadera": lm[mp_pose.PoseLandmark.LEFT_HIP],
            "rodilla": lm[mp_pose.PoseLandmark.LEFT_KNEE],
            "tobillo": lm[mp_pose.PoseLandmark.LEFT_ANKLE],
        },
        "RIGHT": {
            "hombro": lm[mp_pose.PoseLandmark.RIGHT_SHOULDER],
            "codo": lm[mp_pose.PoseLandmark.RIGHT_ELBOW],
            "muneca": lm[mp_pose.PoseLandmark.RIGHT_WRIST],
            "cadera": lm[mp_pose.PoseLandmark.RIGHT_HIP],
            "rodilla": lm[mp_pose.PoseLandmark.RIGHT_KNEE],
            "tobillo": lm[mp_pose.PoseLandmark.RIGHT_ANKLE],
        },
    }

    def score(puntos):
        return sum(getattr(p, "visibility", 0.0) for p in puntos.values())

    lado = max(lados, key=lambda k: score(lados[k]))
    return lados[lado], lado

# ---------------- CÁMARA ----------------
cap = cv2.VideoCapture(0)  # Iriun Cam funciona como webcam

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la camara (indice 0). Verifica Iriun Cam o webcam activa.")

captura_realizada = False
resultado = ""
ultimo_resultado = ""

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        ang_brazos = ang_rodilla = ang_tronco = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark

            # Selecciona automaticamente el lado corporal mejor detectado.
            puntos_lado, lado = seleccionar_lado_mas_visible(lm)

            # Coordenadas
            def xy(p):
                return [int(p.x * w), int(p.y * h)]

            hombro = xy(puntos_lado["hombro"])
            codo = xy(puntos_lado["codo"])
            muneca = xy(puntos_lado["muneca"])
            cadera = xy(puntos_lado["cadera"])
            rodilla = xy(puntos_lado["rodilla"])
            tobillo = xy(puntos_lado["tobillo"])

            # -------- ÁNGULOS --------
            ang_brazos = calcular_angulo(hombro, codo, muneca)
            ang_rodilla = calcular_angulo(cadera, rodilla, tobillo)
            ang_tronco = calcular_angulo(hombro, cadera, rodilla)

            # -------- DIBUJO SIMPLIFICADO --------
            for p in [hombro, codo, muneca, cadera, rodilla, tobillo]:
                cv2.circle(frame, tuple(p), 6, (0, 255, 0), -1)

            cv2.line(frame, tuple(hombro), tuple(codo), (255, 255, 255), 2)
            cv2.line(frame, tuple(codo), tuple(muneca), (255, 255, 255), 2)
            cv2.line(frame, tuple(cadera), tuple(rodilla), (255, 255, 255), 2)
            cv2.line(frame, tuple(rodilla), tuple(tobillo), (255, 255, 255), 2)

            # -------- TEXTO EN PANTALLA --------
            cv2.putText(frame, f"Lado: {lado}", (30, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Brazos: {int(ang_brazos)}", (30, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Rodilla: {int(ang_rodilla)}", (30, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Tronco: {int(ang_tronco)}", (30, 125),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if captura_realizada:
                cv2.putText(frame, resultado, (30, 165),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.putText(frame, "C: evaluar tecnica  |  ESC: salir", (30, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Retroalimentacion Tecnica - Antebrazos", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c') and ang_brazos is not None:
            resultado = evaluar_tecnica(ang_brazos, ang_rodilla, ang_tronco)
            ultimo_resultado = (
                f"{resultado} | Brazos={ang_brazos:.1f}, "
                f"Rodilla={ang_rodilla:.1f}, Tronco={ang_tronco:.1f}"
            )
            print(ultimo_resultado)
            captura_realizada = True
        elif key == ord('c'):
            print("No hay pose valida para evaluar. Colocate lateral a la camara.")

        if key == 27:  # ESC
            break
except KeyboardInterrupt:
    print("Interrupcion manual detectada. Cerrando aplicacion...")
finally:
    cap.release()
    cv2.destroyAllWindows()
