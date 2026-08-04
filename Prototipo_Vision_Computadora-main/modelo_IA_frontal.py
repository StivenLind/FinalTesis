# =========================================================
# MODELO DE IA PARA CLASIFICAR TECNICA FRONTAL DE ANTEBRAZOS
# Features: cabeza, tronco, cadera, rodillas, manos, tobillos
# =========================================================

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

DATASET_PATH = "dataset_frontal_voleibol.csv"
MODEL_PATH = "modelo_tecnica_antebrazos_frontal.pkl"

FEATURES = [
    "dist_manos",
    "ancho_caderas",
    "ancho_rodillas",
    "ancho_tobillos",
    "alineacion_tronco_x",
    "simetria_rodillas_y",
    "simetria_manos_y",
]


def main():
    df = pd.read_csv(DATASET_PATH)

    missing = [c for c in FEATURES + ["clasificacion"] if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en dataset frontal: {missing}")

    x = df[FEATURES]
    y = df["clasificacion"].map({"INCORRECTO": 0, "CORRECTO": 1})

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(max_depth=4, min_samples_leaf=4, random_state=42)),
    ])

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    print("MATRIZ DE CONFUSION")
    print(confusion_matrix(y_test, y_pred))

    print("\nREPORTE DE CLASIFICACION")
    print(classification_report(y_test, y_pred, target_names=["INCORRECTO", "CORRECTO"]))

    cv_folds = min(5, len(df))
    if cv_folds >= 2:
        cv_scores = cross_val_score(model, x, y, cv=cv_folds)
        print(f"\nAccuracy promedio ({cv_folds}-Fold CV):", round(cv_scores.mean(), 3))

    joblib.dump(model, MODEL_PATH)
    print(f"\nModelo frontal guardado como {MODEL_PATH}")


if __name__ == "__main__":
    main()
