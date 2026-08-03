# =========================================================
# MODELO DE IA PARA CLASIFICAR LA TÉCNICA DE ANTEBRAZOS
# Basado en ángulos biomecánicos
# =========================================================

# Requisitos:
# pip install pandas numpy scikit-learn joblib

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import joblib

# ---------------- 1. CARGA DEL DATASET ----------------
df = pd.read_csv("dataset_angulos_voleibol.csv")

# Variables de entrada (features)
X = df[["angulo_brazos", "angulo_rodilla", "angulo_tronco"]]

# Variable objetivo (label)
y = df["clasificacion"].map({"INCORRECTO": 0, "CORRECTO": 1})

# ---------------- 2. DIVISIÓN DE DATOS ----------------
# Debido al tamaño pequeño, se usa una división conservadora
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# ---------------- 3. MODELO ----------------
# Árbol de decisión: interpretable y adecuado para tesis
modelo = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", DecisionTreeClassifier(
        max_depth=3,
        min_samples_leaf=3,
        random_state=42
    ))
])

# ---------------- 4. ENTRENAMIENTO ----------------
modelo.fit(X_train, y_train)

# ---------------- 5. EVALUACIÓN ----------------
y_pred = modelo.predict(X_test)

print("MATRIZ DE CONFUSIÓN")
print(confusion_matrix(y_test, y_pred))

print("\nREPORTE DE CLASIFICACIÓN")
print(classification_report(y_test, y_pred, target_names=["INCORRECTO", "CORRECTO"]))

# ---------------- 6. VALIDACIÓN CRUZADA ----------------
cv_scores = cross_val_score(modelo, X, y, cv=5)
print("\nAccuracy promedio (5-Fold CV):", round(cv_scores.mean(), 3))

# ---------------- 7. GUARDAR MODELO ----------------
joblib.dump(modelo, "modelo_tecnica_antebrazos.pkl")
print("\nModelo guardado como modelo_tecnica_antebrazos.pkl")

# ---------------- 8. FUNCIÓN DE PREDICCIÓN ----------------
def predecir_tecnica(angulo_brazos, angulo_rodilla, angulo_tronco):
    # Mantener nombres de columnas evita warnings de scikit-learn.
    entrada = pd.DataFrame([{
        "angulo_brazos": angulo_brazos,
        "angulo_rodilla": angulo_rodilla,
        "angulo_tronco": angulo_tronco
    }])
    pred = modelo.predict(entrada)[0]
    return "CORRECTO" if pred == 1 else "INCORRECTO"

# Ejemplo de uso
print("\nEjemplo de predicción:")
print(predecir_tecnica(170, 120, 135))
