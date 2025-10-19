#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementación de Estimación de Profundidad en Tiempo Real con Depth Anything V2 y OpenCV

Requisitos:
- Python 3.9+ recomendado
- PyTorch, torchvision, timm, opencv-python, numpy, transformers, pillow, huggingface_hub

Notas:
- Este script NO incluye los pesos. Debes descargarlos y colocarlos en ./models/ (ver README.md).
- Compatibilidad CPU/GPU (CUDA) automática.
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

import torch

# Usamos Transformers para cargar Depth Anything V2 (clases nativas)
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

def find_model_dir(root_models: str) -> str:
    """
    Busca un directorio de modelo válido dentro de la carpeta models/.
    Recomendado: 'Depth-Anything-V2-Small-hf' o 'Depth-Anything-V2-Base-hf' descargado desde Hugging Face.
    Retorna la ruta del modelo si lo encuentra, o levanta FileNotFoundError.
    """
    if not os.path.isdir(root_models):
        raise FileNotFoundError(f"No existe la carpeta de modelos: {root_models}")

    # Candidatos comunes
    candidates = [
        "Depth-Anything-V2-Small-hf",
        "Depth-Anything-V2-Base-hf",
        "Depth-Anything-V2-Large-hf",
        "depth-anything-v2-small-hf",
        "depth-anything-v2-base-hf",
        "depth-anything-v2-large-hf",
    ]

    # 1) Checar candidatos conocidos
    for c in candidates:
        d = os.path.join(root_models, c)
        if os.path.isdir(d):
            # Validación superficial: presencia de config.json y model.safetensors / pytorch_model.bin
            config_ok = os.path.isfile(os.path.join(d, "config.json"))
            weight_ok = any(os.path.isfile(os.path.join(d, name)) for name in [
                "pytorch_model.bin", "model.safetensors", "pytorch_model-00001-of-00002.bin"
            ])
            if config_ok and weight_ok:
                return d

    # 2) Si no, intentar el primero que tenga config + pesos
    for entry in os.listdir(root_models):
        d = os.path.join(root_models, entry)
        if os.path.isdir(d):
            config_ok = os.path.isfile(os.path.join(d, "config.json"))
            weight_ok = any(os.path.isfile(os.path.join(d, name)) for name in [
                "pytorch_model.bin", "model.safetensors", "pytorch_model-00001-of-00002.bin"
            ])
            if config_ok and weight_ok:
                return d

    raise FileNotFoundError(
        "No se encontró un directorio de modelo válido dentro de 'models/'.\n"
        "Asegúrate de descargar, por ejemplo, 'depth-anything/Depth-Anything-V2-Small-hf' "
        "desde Hugging Face y colocarlo en ./models/Depth-Anything-V2-Small-hf"
    )


def normalize_depth_to_uint8(depth_np: np.ndarray) -> np.ndarray:
    """
    Normaliza un mapa de profundidad a [0, 255] en uint8 para visualización.
    Maneja casos degenerados (valores constantes, NaNs, infs).
    """
    d = depth_np.copy()
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    d_min, d_max = float(d.min()), float(d.max())
    if d_max - d_min < 1e-6:
        return np.zeros_like(d, dtype=np.uint8)
    d_norm = (d - d_min) / (d_max - d_min)
    d_uint8 = (d_norm * 255.0).clip(0, 255).astype(np.uint8)
    return d_uint8


def main():
    parser = argparse.ArgumentParser(description="Estimación de profundidad en tiempo real con Depth Anything V2 + OpenCV.")
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara (default: 0).")
    parser.add_argument("--models_dir", type=str, default="models", help="Carpeta de modelos (default: ./models).")
    parser.add_argument("--width", type=int, default=None, help="Redimensionar ancho de frame para inferencia (opcional).")
    parser.add_argument("--height", type=int, default=None, help="Redimensionar alto de frame para inferencia (opcional).")
    parser.add_argument("--fp16", action="store_true", help="Forzar FP16 si hay CUDA (por defecto se decide automáticamente).")
    args = parser.parse_args()

    # Selección de dispositivo
    has_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if has_cuda else "cpu")
    print(f"[INFO] Dispositivo: {device} (CUDA disponible: {has_cuda})")

    # Cargar modelo
    try:
        model_dir = find_model_dir(args.models_dir)
        print(f"[INFO] Usando modelo en: {model_dir}")
        dtype = torch.float16 if (has_cuda and args.fp16) else (torch.float16 if has_cuda else torch.float32)

        # Image processor y modelo (local_files_only=True para forzar uso desde ./models)
        processor = AutoImageProcessor.from_pretrained(model_dir, local_files_only=True)
        model = AutoModelForDepthEstimation.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            local_files_only=True
        ).to(device)
        model.eval()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        print("\nSigue las instrucciones del README.md para descargar el modelo y colocarlo en ./models/\n")
        sys.exit(1)
    except OSError as e:
        print(f"[ERROR] No se pudo cargar el modelo desde {args.models_dir}: {e}")
        print("¿Colocaste los archivos correctos (config.json, pesos *.bin o *.safetensors) en la carpeta de modelo?")
        sys.exit(1)

    # Inicializar cámara
    cap = cv2.VideoCapture(args.camera, cv2.CAP_ANY)
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir la cámara con índice {args.camera}. Verifica que exista y no esté en uso.")
        sys.exit(1)

    # Opcional: configurar resolución de captura si se solicitó
    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Nombres de ventanas
    win_original = "Webcam - Original (q para salir)"
    win_depth = "Depth Anything V2 - Profundidad"

    cv2.namedWindow(win_original, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_depth, cv2.WINDOW_NORMAL)

    print("[INFO] Presiona 'q' en cualquiera de las ventanas para finalizar.")

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("[WARN] Fallo al leer frame de la cámara. Intentando continuar...")
                time.sleep(0.01)
                continue

            # Mostrar original
            cv2.imshow(win_original, frame_bgr)

            # Preprocesamiento: BGR -> RGB
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # Procesar con el image processor (acepta numpy/PIL)
            inputs = processor(images=frame_rgb, return_tensors="pt")
            # Mover tensores a dispositivo
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = model(**inputs)
                # La salida estándar para depth es 'predicted_depth' (B, H, W)
                pred = outputs.predicted_depth

                # Reescalar a tamaño del frame original para visualización
                # shape: [B, 1, H, W] o [B, H, W] según versión
                if pred.ndim == 3:
                    pred = pred.unsqueeze(1)  # [B,1,H,W]
                pred_resized = torch.nn.functional.interpolate(
                    pred,
                    size=(frame_bgr.shape[0], frame_bgr.shape[1]),
                    mode="bilinear",
                    align_corners=False
                )

                depth_np = pred_resized[0, 0].detach().float().cpu().numpy()
                depth_u8 = normalize_depth_to_uint8(depth_np)

                # Colorizar (Inferno)
                depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

            cv2.imshow(win_depth, depth_color)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupción por teclado. Cerrando...")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
