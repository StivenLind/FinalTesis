"""Aumenta imágenes del dataset con transformaciones suaves usando Albumentations.

Uso rápido:
    python aumentar_dataset_albumentations.py

Ejemplo con parámetros:
    python aumentar_dataset_albumentations.py --dataset-dir "Entrenamiento Data Set" --num-aug 8 --max-rotate 5
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import albumentations as A
import cv2
import numpy as np


EXTENSIONES_IMAGEN = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def crear_transformaciones(max_rotate: int = 5) -> A.Compose:
    """Crea pipeline de aumentos suaves que preservan la biomecánica."""
    return A.Compose(
        [
            A.RandomBrightnessContrast(
                brightness_limit=0.15,
                contrast_limit=0.15,
                p=0.7,
            ),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
            A.GaussNoise(std_range=(0.01, 0.03), mean_range=(0.0, 0.0), p=0.35),
            A.Rotate(
                limit=(-max_rotate, max_rotate),
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.45,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.02,
                scale_limit=0.04,
                rotate_limit=0,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.35,
            ),
            A.RandomGamma(gamma_limit=(90, 110), p=0.45),
        ]
    )


def es_imagen(path: Path) -> bool:
    return path.suffix.lower() in EXTENSIONES_IMAGEN


def es_aug_generada(path: Path) -> bool:
    return "_aug" in path.stem.lower()


def leer_imagen(path: Path) -> np.ndarray | None:
    """Lee imagen con soporte para rutas con espacios y caracteres especiales en Windows."""
    try:
        datos = np.fromfile(str(path), dtype=np.uint8)
        if datos.size == 0:
            return None
        return cv2.imdecode(datos, cv2.IMREAD_COLOR)
    except Exception:
        return None


def guardar_imagen(path: Path, image: np.ndarray) -> bool:
    """Guarda imagen en disco evitando problemas de codificación de ruta."""
    extension = path.suffix.lower()
    if extension in {".jpg", ".jpeg"}:
        ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    elif extension == ".png":
        ok, encoded = cv2.imencode(".png", image, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
    elif extension == ".webp":
        ok, encoded = cv2.imencode(".webp", image, [int(cv2.IMWRITE_WEBP_QUALITY), 95])
    else:
        ok, encoded = cv2.imencode(extension, image)

    if not ok:
        return False

    try:
        encoded.tofile(str(path))
        return True
    except Exception:
        return False


def iterar_imagenes_originales(dataset_dir: Path) -> Iterable[Path]:
    for path in dataset_dir.rglob("*"):
        if path.is_file() and es_imagen(path) and not es_aug_generada(path):
            yield path


def siguiente_indice_disponible(image_path: Path) -> int:
    base = image_path.stem
    ext = image_path.suffix
    idx = 1
    while True:
        candidate = image_path.with_name(f"{base}_aug{idx}{ext}")
        if not candidate.exists():
            return idx
        idx += 1


def procesar_imagen(
    image_path: Path,
    transform: A.Compose,
    num_aug: int,
) -> int:
    image = leer_imagen(image_path)
    if image is None:
        print(f"  [ADVERTENCIA] No se pudo leer: {image_path}")
        return 0

    generadas = 0
    idx = siguiente_indice_disponible(image_path)

    for _ in range(num_aug):
        aug = transform(image=image)["image"]
        out_path = image_path.with_name(f"{image_path.stem}_aug{idx}{image_path.suffix}")
        if guardar_imagen(out_path, aug):
            generadas += 1
        else:
            print(f"  [ADVERTENCIA] No se pudo guardar: {out_path}")
        idx += 1

    return generadas


def procesar_dataset(dataset_dir: Path, num_aug: int, max_rotate: int) -> None:
    transform = crear_transformaciones(max_rotate=max_rotate)

    originales = sorted(iterar_imagenes_originales(dataset_dir))
    total_originales = len(originales)

    print(f"Imágenes originales encontradas: {total_originales}")
    if total_originales == 0:
        print("No se encontraron imágenes para procesar.")
        return

    total_generadas = 0
    procesadas = 0

    for i, image_path in enumerate(originales, start=1):
        nuevas = procesar_imagen(image_path, transform=transform, num_aug=num_aug)
        total_generadas += nuevas
        procesadas += 1
        print(f"Imagen {i}/{total_originales} procesada: +{nuevas}")

    total_dataset = total_originales + total_generadas

    print("\nResumen")
    print(f"Originales: {total_originales}")
    print(f"Procesadas: {procesadas}")
    print(f"Generadas: {total_generadas}")
    print(f"Total del dataset: {total_dataset}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aumenta dataset de imágenes con transformaciones suaves de Albumentations."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("Entrenamiento Data Set"),
        help="Directorio raíz del dataset de imágenes.",
    )
    parser.add_argument(
        "--num-aug",
        type=int,
        default=7,
        help="Cantidad de imágenes nuevas por cada imagen original (recomendado: 6-8).",
    )
    parser.add_argument(
        "--max-rotate",
        type=int,
        default=5,
        help="Rotación máxima en grados para Rotate (recomendado: 5; máximo 8).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_aug <= 0:
        raise ValueError("--num-aug debe ser mayor que 0.")
    if args.max_rotate < 0 or args.max_rotate > 8:
        raise ValueError("--max-rotate debe estar entre 0 y 8 para preservar la postura.")
    if not args.dataset_dir.exists() or not args.dataset_dir.is_dir():
        raise FileNotFoundError(f"No existe el directorio de dataset: {args.dataset_dir}")

    procesar_dataset(args.dataset_dir, num_aug=args.num_aug, max_rotate=args.max_rotate)


if __name__ == "__main__":
    main()
