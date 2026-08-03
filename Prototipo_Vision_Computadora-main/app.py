import base64
import os
import time
from threading import Lock
from datetime import datetime

import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
MODEL_PATH = os.path.join(BASE_DIR, "modelo_tecnica_antebrazos.pkl")
MODEL_PATH_FRONTAL = os.path.join(BASE_DIR, "modelo_tecnica_antebrazos_frontal.pkl")
PROCESS_WIDTH = 320
PROCESS_HEIGHT = 240

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"No se encontro el modelo en: {MODEL_PATH}. Ejecuta primero modelo_IA.py"
    )

model = joblib.load(MODEL_PATH)
model_frontal = joblib.load(MODEL_PATH_FRONTAL) if os.path.exists(MODEL_PATH_FRONTAL) else None

BaseOptions = python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

import urllib.request
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker_lite.task")
if not os.path.exists(POSE_MODEL_PATH):
    print("Downloading MediaPipe Pose model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        POSE_MODEL_PATH
    )
base_options = BaseOptions(model_asset_path=POSE_MODEL_PATH)

options = PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=VisionRunningMode.VIDEO,
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_segmentation_masks=False)

_timestamp_lock = Lock()
_last_timestamp_ms = 0
_landmarker_lock = Lock()
_pose_landmarker = None
_view_lock = Lock()
_view_state = {
    "stable_view": None,
    "candidate_view": None,
    "candidate_count": 0,
}
VIEW_SWITCH_CONFIRM_FRAMES = 4
VIEW_SWITCH_MARGIN = 0.12

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")


def get_pose_landmarker():
    global _pose_landmarker
    if _pose_landmarker is not None:
        return _pose_landmarker

    with _landmarker_lock:
        if _pose_landmarker is None:
            _pose_landmarker = PoseLandmarker.create_from_options(options)
    return _pose_landmarker


def get_next_timestamp_ms():
    global _last_timestamp_ms
    current_ms = int(time.monotonic() * 1000)
    with _timestamp_lock:
        if current_ms <= _last_timestamp_ms:
            current_ms = _last_timestamp_ms + 1
        _last_timestamp_ms = current_ms
    return current_ms


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    cos_angle = np.dot(ba, bc) / denom
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def select_visible_side(landmarks):
    # Landmark indices for new MediaPipe Vision API (same as old)
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    LEFT_HIP = 23
    LEFT_KNEE = 25
    LEFT_ANKLE = 27
    RIGHT_SHOULDER = 12
    RIGHT_ELBOW = 14
    RIGHT_WRIST = 16
    RIGHT_HIP = 24
    RIGHT_KNEE = 26
    RIGHT_ANKLE = 28

    sides = {
        "LEFT": {
            "hombro": landmarks[LEFT_SHOULDER],
            "codo": landmarks[LEFT_ELBOW],
            "muneca": landmarks[LEFT_WRIST],
            "cadera": landmarks[LEFT_HIP],
            "rodilla": landmarks[LEFT_KNEE],
            "tobillo": landmarks[LEFT_ANKLE],
        },
        "RIGHT": {
            "hombro": landmarks[RIGHT_SHOULDER],
            "codo": landmarks[RIGHT_ELBOW],
            "muneca": landmarks[RIGHT_WRIST],
            "cadera": landmarks[RIGHT_HIP],
            "rodilla": landmarks[RIGHT_KNEE],
            "tobillo": landmarks[RIGHT_ANKLE],
        },
    }

    def score(points):
        return sum(getattr(p, "visibility", 0.0) for p in points.values())

    selected_side = max(sides, key=lambda s: score(sides[s]))
    return selected_side, sides[selected_side]


def feedback_messages(angles, result):
    feedback = []

    if angles["brazos"] < 150:
        feedback.append("Extiende mas los brazos para formar una plataforma firme.")
    elif angles["brazos"] > 180:
        feedback.append("Relaja ligeramente los codos para evitar hiperextension.")

    if angles["rodilla"] < 90:
        feedback.append("Sube un poco la postura: la flexion de rodilla es muy cerrada.")
    elif angles["rodilla"] > 135:
        feedback.append("Flexiona mas las rodillas para estabilizar el centro de gravedad.")

    if angles["tronco"] < 110:
        feedback.append("Endereza un poco el tronco, hay demasiada inclinacion.")
    elif angles["tronco"] > 160:
        feedback.append("Inclina un poco mas el tronco para una mejor recepcion.")

    if result == "CORRECTO" and not feedback:
        feedback.append("Tecnica correcta. Mantener postura y sincronizacion.")
    elif result == "CORRECTO":
        feedback.append("Buena ejecucion general. Ajusta detalles para mayor precision.")

    if result == "INCORRECTO" and not feedback:
        feedback.append("Se detectaron errores tecnicos. Ajusta postura y repite.")

    return feedback


def _distance_xy(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def compute_frontal_features(landmarks):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28

    def point_xy(idx):
        p = landmarks[idx]
        return [float(p.x), float(p.y)]

    hombro_izq = point_xy(LEFT_SHOULDER)
    hombro_der = point_xy(RIGHT_SHOULDER)
    codo_izq = point_xy(LEFT_ELBOW)
    codo_der = point_xy(RIGHT_ELBOW)
    muneca_izq = point_xy(LEFT_WRIST)
    muneca_der = point_xy(RIGHT_WRIST)
    cadera_izq = point_xy(LEFT_HIP)
    cadera_der = point_xy(RIGHT_HIP)
    rodilla_izq = point_xy(LEFT_KNEE)
    rodilla_der = point_xy(RIGHT_KNEE)
    tobillo_izq = point_xy(LEFT_ANKLE)
    tobillo_der = point_xy(RIGHT_ANKLE)

    tronco = [
        (hombro_izq[0] + hombro_der[0]) / 2.0,
        (hombro_izq[1] + hombro_der[1]) / 2.0,
    ]
    cadera = [
        (cadera_izq[0] + cadera_der[0]) / 2.0,
        (cadera_izq[1] + cadera_der[1]) / 2.0,
    ]
    ancho_hombros = _distance_xy(hombro_izq, hombro_der)
    ancho_caderas = _distance_xy(cadera_izq, cadera_der)
    ancho_rodillas = _distance_xy(rodilla_izq, rodilla_der)
    ancho_tobillos = _distance_xy(tobillo_izq, tobillo_der)
    dist_manos = _distance_xy(muneca_izq, muneca_der)

    trunk_center_offset_x = abs(tronco[0] - cadera[0])
    simetria_rodillas_y = abs(rodilla_izq[1] - rodilla_der[1])
    simetria_manos_y = abs(muneca_izq[1] - muneca_der[1])

    # Normalizaciones para estabilidad ante cambios de escala.
    scale = max(ancho_hombros, 1e-6)
    features = {
        "dist_manos": dist_manos / scale,
        "ancho_caderas": ancho_caderas / scale,
        "ancho_rodillas": ancho_rodillas / scale,
        "ancho_tobillos": ancho_tobillos / scale,
        "alineacion_tronco_x": trunk_center_offset_x / scale,
        "simetria_rodillas_y": simetria_rodillas_y / scale,
        "simetria_manos_y": simetria_manos_y / scale,
    }

    points = {
        "hombro_izq": hombro_izq,
        "hombro_der": hombro_der,
        "codo_izq": codo_izq,
        "codo_der": codo_der,
        "muneca_izq": muneca_izq,
        "muneca_der": muneca_der,
        "cadera_izq": cadera_izq,
        "cadera_der": cadera_der,
        "rodilla_izq": rodilla_izq,
        "rodilla_der": rodilla_der,
        "tobillo_izq": tobillo_izq,
        "tobillo_der": tobillo_der,
    }

    return points, features


def compute_frontal_angles(points):
    angulo_brazo_izq = calculate_angle(points["hombro_izq"], points["codo_izq"], points["muneca_izq"])
    angulo_brazo_der = calculate_angle(points["hombro_der"], points["codo_der"], points["muneca_der"])
    angulo_rodilla_izq = calculate_angle(points["cadera_izq"], points["rodilla_izq"], points["tobillo_izq"])
    angulo_rodilla_der = calculate_angle(points["cadera_der"], points["rodilla_der"], points["tobillo_der"])
    angulo_tronco_izq = calculate_angle(points["hombro_izq"], points["cadera_izq"], points["rodilla_izq"])
    angulo_tronco_der = calculate_angle(points["hombro_der"], points["cadera_der"], points["rodilla_der"])

    return {
        "brazos": round((angulo_brazo_izq + angulo_brazo_der) / 2.0, 1),
        "rodilla": round((angulo_rodilla_izq + angulo_rodilla_der) / 2.0, 1),
        "tronco": round((angulo_tronco_izq + angulo_tronco_der) / 2.0, 1),
        "brazo_izq": round(angulo_brazo_izq, 1),
        "brazo_der": round(angulo_brazo_der, 1),
        "rodilla_izq": round(angulo_rodilla_izq, 1),
        "rodilla_der": round(angulo_rodilla_der, 1),
    }


def frontal_feedback(metrics, result):
    feedback = []

    if metrics["alineacion_tronco_x"] > 0.12:
        feedback.append("Centra el tronco sobre la cadera para mejorar estabilidad.")
    if metrics["simetria_rodillas_y"] > 0.08:
        feedback.append("Alinea la altura de ambas rodillas para una base simetrica.")
    if metrics["simetria_manos_y"] > 0.08:
        feedback.append("Mantén ambas manos a una altura similar al formar la plataforma.")
    if metrics["ancho_tobillos"] < 0.5:
        feedback.append("Abre un poco mas la base de apoyo con los tobillos.")

    if result == "CORRECTO" and not feedback:
        feedback.append("Tecnica frontal correcta. Mantener alineacion y control postural.")
    elif result == "INCORRECTO" and not feedback:
        feedback.append("Ajusta simetria corporal y base de apoyo para mejorar la tecnica.")

    return feedback


def classify_frontal(features):
    if model_frontal is not None:
        model_input = pd.DataFrame([features])
        prediction = int(model_frontal.predict(model_input)[0])
        return "CORRECTO" if prediction == 1 else "INCORRECTO"

    rules_ok = 0
    rules_ok += 1 if features["alineacion_tronco_x"] <= 0.12 else 0
    rules_ok += 1 if features["simetria_rodillas_y"] <= 0.08 else 0
    rules_ok += 1 if features["simetria_manos_y"] <= 0.08 else 0
    rules_ok += 1 if 0.5 <= features["ancho_tobillos"] <= 1.5 else 0
    return "CORRECTO" if rules_ok >= 3 else "INCORRECTO"


def denormalize_point(point, width, height):
    return [int(point[0] * width), int(point[1] * height)]


def infer_view_from_landmarks(landmarks):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    shoulder_width = abs(ls.x - rs.x)
    hip_width = abs(lh.x - rh.x)
    shoulder_depth_diff = abs(getattr(ls, "z", 0.0) - getattr(rs, "z", 0.0))

    frontal_score = 0
    frontal_score += 1 if shoulder_width >= 0.14 else 0
    frontal_score += 1 if hip_width >= 0.11 else 0
    frontal_score += 1 if shoulder_depth_diff <= 0.12 else 0

    return "frontal" if frontal_score >= 2 else "lateral"


def infer_stable_view_from_landmarks(landmarks):
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    ls = landmarks[LEFT_SHOULDER]
    rs = landmarks[RIGHT_SHOULDER]
    lh = landmarks[LEFT_HIP]
    rh = landmarks[RIGHT_HIP]

    shoulder_width = abs(ls.x - rs.x)
    hip_width = abs(lh.x - rh.x)
    shoulder_depth_diff = abs(getattr(ls, "z", 0.0) - getattr(rs, "z", 0.0))

    frontal_strength = (
        np.clip((shoulder_width - 0.11) / 0.12, 0.0, 1.0)
        + np.clip((hip_width - 0.09) / 0.12, 0.0, 1.0)
        + np.clip((0.18 - shoulder_depth_diff) / 0.18, 0.0, 1.0)
    ) / 3.0

    lateral_strength = (
        np.clip((0.16 - shoulder_width) / 0.16, 0.0, 1.0)
        + np.clip((0.14 - hip_width) / 0.14, 0.0, 1.0)
        + np.clip((shoulder_depth_diff - 0.05) / 0.20, 0.0, 1.0)
    ) / 3.0

    score_diff = float(frontal_strength - lateral_strength)
    raw_view = "frontal" if score_diff >= 0 else "lateral"

    with _view_lock:
        stable_view = _view_state["stable_view"]

        if stable_view is None:
            _view_state["stable_view"] = raw_view
            _view_state["candidate_view"] = None
            _view_state["candidate_count"] = 0
            return raw_view

        if raw_view == stable_view:
            _view_state["candidate_view"] = None
            _view_state["candidate_count"] = 0
            return stable_view

        if abs(score_diff) < VIEW_SWITCH_MARGIN:
            _view_state["candidate_view"] = None
            _view_state["candidate_count"] = 0
            return stable_view

        if _view_state["candidate_view"] == raw_view:
            _view_state["candidate_count"] += 1
        else:
            _view_state["candidate_view"] = raw_view
            _view_state["candidate_count"] = 1

        if _view_state["candidate_count"] >= VIEW_SWITCH_CONFIRM_FRAMES:
            _view_state["stable_view"] = raw_view
            _view_state["candidate_view"] = None
            _view_state["candidate_count"] = 0

        return _view_state["stable_view"]


def parse_image_from_data_url(data_url):
    if not data_url:
        return None

    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    image_bytes = base64.b64decode(encoded)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame


@app.after_request
def add_cors_headers(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/api/evaluar_frame", methods=["POST", "OPTIONS"])
def evaluate_frame():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    vista = str(payload.get("vista", "auto")).lower().strip()
    frame = parse_image_from_data_url(payload.get("image"))

    if frame is None:
        return jsonify({"ok": False, "message": "Frame invalido."}), 400

    original_height, original_width, _ = frame.shape

    frame_small = cv2.resize(
        frame,
        (PROCESS_WIDTH, PROCESS_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    frame_rgb = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    frame_timestamp_ms = get_next_timestamp_ms()
    pose_landmarker = get_pose_landmarker()
    results = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)

    if not results.pose_landmarks or len(results.pose_landmarks) == 0:
        return jsonify({
            "ok": False,
            "message": "No se detecto pose. Ajusta tu posicion frente a la camara.",
            "timestamp": datetime.utcnow().isoformat()
        })

    landmarks = results.pose_landmarks[0]
    if vista not in {"frontal", "lateral"}:
        vista = infer_stable_view_from_landmarks(landmarks)

    if vista == "frontal":
        frontal_points_norm, frontal_features = compute_frontal_features(landmarks)
        frontal_angles = compute_frontal_angles(frontal_points_norm)
        frontal_points = {
            name: denormalize_point(point, original_width, original_height)
            for name, point in frontal_points_norm.items()
        }
        frontal_result = classify_frontal(frontal_features)
        return jsonify({
            "ok": True,
            "vista": "frontal",
            "resultado": frontal_result,
            "angulos": frontal_angles,
            "metricas_frontal": {k: round(v, 3) for k, v in frontal_features.items()},
            "feedback": frontal_feedback(frontal_features, frontal_result),
            "puntos_frontal": frontal_points,
            "modelo_frontal_cargado": model_frontal is not None,
            "timestamp": datetime.utcnow().isoformat()
        })

    side_name, side_points = select_visible_side(landmarks)

    scale_x = original_width / PROCESS_WIDTH
    scale_y = original_height / PROCESS_HEIGHT

    def to_xy(point):
        return [
            int(point.x * PROCESS_WIDTH * scale_x),
            int(point.y * PROCESS_HEIGHT * scale_y),
        ]

    hombro = to_xy(side_points["hombro"])
    codo = to_xy(side_points["codo"])
    muneca = to_xy(side_points["muneca"])
    cadera = to_xy(side_points["cadera"])
    rodilla = to_xy(side_points["rodilla"])
    tobillo = to_xy(side_points["tobillo"])

    angles = {
        "brazos": round(calculate_angle(hombro, codo, muneca), 1),
        "rodilla": round(calculate_angle(cadera, rodilla, tobillo), 1),
        "tronco": round(calculate_angle(hombro, cadera, rodilla), 1),
    }

    model_input = pd.DataFrame([{
        "angulo_brazos": angles["brazos"],
        "angulo_rodilla": angles["rodilla"],
        "angulo_tronco": angles["tronco"],
    }])

    prediction = int(model.predict(model_input)[0])
    result = "CORRECTO" if prediction == 1 else "INCORRECTO"

    return jsonify({
        "ok": True,
        "vista": "lateral",
        "resultado": result,
        "lado": side_name,
        "angulos": angles,
        "feedback": feedback_messages(angles, result),
        "puntos": {
            "hombro": hombro,
            "codo": codo,
            "muneca": muneca,
            "cadera": cadera,
            "rodilla": rodilla,
            "tobillo": tobillo,
        },
        "timestamp": datetime.utcnow().isoformat()
    })


@app.route("/")
def root():
    return send_from_directory(FRONTEND_DIR, "index.html")


@app.route("/<path:filename>")
def serve_frontend(filename):
    return send_from_directory(FRONTEND_DIR, filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
