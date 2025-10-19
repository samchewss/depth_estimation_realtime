 # Implementación de Estimación de Profundidad en Tiempo Real con Depth Anything V2 y OpenCV

**EQUIPO: Samantha Elizabeth Chew Arenas y Alonso Pérez Medrano**

Este proyecto muestra cómo ejecutar **estimación de profundidad monocular en tiempo real** usando **Depth Anything V2** (DAv2) con **OpenCV** para capturar la webcam y visualizar simultáneamente la imagen original y el mapa de profundidad colorizado.

> **Nota:** El repositorio y pesos del modelo **NO** están incluidos. A continuación se explica cómo descargarlos y colocarlos en `models/`.

---

## Objetivo
Implementar un script en Python (`main.py`) que:
- Capture video de la webcam con `cv2.VideoCapture(0)`.
- Efectúe inferencia de profundidad por frame con **Depth Anything V2**.
- Muestre dos ventanas en tiempo real: **Original** y **Profundidad** (colorizada con `COLORMAP_INFERNO`).
- Funcione tanto en **CPU** como en **GPU** (si hay CUDA, se usa automáticamente).

---

## Estructura del proyecto
```
depth_estimation_realtime/
├── main.py
├── requirements.txt
├── README.md
└── models/                 # Coloca aquí el modelo pre-entrenado (vacío por defecto)
```

---

## Entorno y dependencias

1. **Crear entorno virtual (recomendado)**

En Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

En macOS / Linux (bash/zsh):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. **Instalar dependencias**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> Incluye: `torch`, `torchvision`, `opencv-python`, `numpy`, `timm`, `transformers`, `pillow`, `huggingface_hub`.

---

## Descarga del modelo pre-entrenado (NO incluido)

Usaremos los pesos publicados en **Hugging Face** por el equipo de Depth Anything V2. Se recomiendan los modelos **Small** (rápido) o **Base** (mejor detalle).

- **Small:** `depth-anything/Depth-Anything-V2-Small-hf`  
- **Base:** `depth-anything/Depth-Anything-V2-Base-hf`

### Opción A) Descarga automática con `huggingface_hub`

Con el entorno activado, ejecuta uno de los siguientes comandos para descargar en la carpeta `models/`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
import os
target = os.path.join("models", "Depth-Anything-V2-Small-hf")  # o "Depth-Anything-V2-Base-hf"
snapshot_download(repo_id="depth-anything/Depth-Anything-V2-Small-hf", local_dir=target, local_dir_use_symlinks=False)
print("Descarga completa en:", target)
PY
```

Para el **Base**:
```bash
python - <<'PY'
from huggingface_hub import snapshot_download
import os
target = os.path.join("models", "Depth-Anything-V2-Base-hf")
snapshot_download(repo_id="depth-anything/Depth-Anything-V2-Base-hf", local_dir=target, local_dir_use_symlinks=False)
print("Descarga completa en:", target)
PY
```

> Si usas un **token** de Hugging Face, autentícate con `huggingface-cli login` antes.

### Opción B) Descarga manual (navegador)
1. Abre el modelo en Hugging Face y descarga todos los archivos (por ejemplo, `config.json`, `model.safetensors` o `pytorch_model.bin`, etc.).  
2. Crea una carpeta dentro de `models/` con el **mismo nombre** del repositorio, por ejemplo:
   - `models/Depth-Anything-V2-Small-hf/`
   - `models/Depth-Anything-V2-Base-hf/`
3. Copia ahí todos los archivos descargados.

---

## Ejecución

Con el entorno activado y el modelo ya descargado en `./models/`:

```bash
python main.py
```

Parámetros opcionales:
```bash
python main.py --camera 0 --width 1280 --height 720 --fp16
```
- `--camera`: índice de la webcam (default: 0).
- `--width` / `--height`: redimensiona la captura (útil para controlar la carga).
- `--fp16`: fuerza FP16 en GPU (si hay CUDA).

**Salida esperada:** Se abrirán dos ventanas:  
- **Webcam - Original (q para salir)**
- **Depth Anything V2 - Profundidad** (colorizada con Inferno)

Presiona **`q`** para cerrar.

---

## Consejos de rendimiento y limitaciones
- **Modelo Small** suele ser el más adecuado para *tiempo real* en CPU. En GPU, **Base** puede ir fluido.
- Ajusta la resolución de la cámara con `--width/--height` para mantener FPS estables.
- En CPU, la latencia aumentará con resoluciones muy altas.
- La estimación es **monocular**: las profundidades son relativas; pueden invertirse según calibración/escena.
- Iluminación, texturas repetitivas o escenas con poca textura pueden afectar el resultado.

---

## Fuentes y referencias
- Repositorio oficial: Depth Anything V2 (GitHub).  
- Modelos en Hugging Face: *Depth-Anything-V2-Small-hf*, *Depth-Anything-V2-Base-hf*.  
- Documentación Transformers para `DepthAnythingV2`.

---

## Solución de problemas (FAQ)

**La cámara no abre**  
- Verifica el índice `--camera` o cierra otras apps que usen la webcam.
- Reinstala `opencv-python` si es necesario.

**Error: modelo no encontrado**  
- Revisa que `models/` contenga una subcarpeta con los archivos (`config.json` y pesos) del modelo.
- Usa los scripts de descarga anteriores (Opción A).

**CUDA no detectada**  
- El script funcionará en CPU automáticamente. Para GPU, instala PyTorch con CUDA adecuado para tu sistema (consulta la web de PyTorch).
