# IA Voleibol - Sistema de Análisis de Técnica de Antebrazos

Proyecto de investigación que utiliza visión por computadora e inteligencia artificial para analizar y mejorar la técnica de golpeo con antebrazos en jugadores de voleibol.

## 📁 Estructura del Proyecto

```
Prototipo_Vision_Computadora/
├── backend/                    # Servidor Flask y modelos ML
│   ├── app.py                 # Servidor principal
│   ├── modelo_*.pkl           # Modelos entrenados
│   ├── medir_*.py             # Scripts para generar datasets
│   ├── modelo_IA*.py          # Scripts para entrenar modelos
│   ├── capturas/              # Carpeta de capturas guardadas
│   └── README.md              # Documentación del backend
│
├── frontend/                   # Interfaz web
│   ├── biomecanica.html       # Página de entrenamiento en tiempo real
│   ├── index.html             # Página de inicio
│   ├── styles.css             # Estilos generales
│   └── images/                # Imágenes del proyecto
│
├── run.py                      # Script principal para ejecutar servidor
├── INICIAR_SERVIDOR.bat        # Atajo para Windows
├── iniciar_servidor.sh         # Atajo para Linux/macOS
├── requirements.txt            # Dependencias del proyecto
└── README.md                   # Este archivo
```

## 🚀 Inicio Rápido

### Opción 1: Windows (Recomendado)
Haz doble clic en **`INICIAR_SERVIDOR.bat`**

### Opción 2: Terminal (Cualquier Sistema)
```bash
# Desde la raíz del proyecto
python run.py
```

El servidor estará disponible en: **http://127.0.0.1:5000**

## 📋 Requisitos

- Python 3.8+
- pip (gestor de paquetes Python)
- Cámara web conectada

## 🔧 Instalación

1. **Clonar o descargar el proyecto**
   ```bash
   cd Prototipo_Vision_Computadora
   ```

2. **Crear entorno virtual** (si no existe)
   ```bash
   python -m venv .venv
   ```

3. **Activar entorno virtual**
   - Windows: `.venv\Scripts\activate.bat`
   - Linux/macOS: `source .venv/bin/activate`

4. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

5. **Ejecutar el servidor**
   ```bash
   python run.py
   ```

## 📚 Funcionalidades

### Análisis en Tiempo Real
- Detección automática de postura frontal y lateral
- Cálculo de ángulos biomecánicos (brazos, rodilla, tronco)
- Clasificación de técnica (CORRECTO/INCORRECTO)
- Retroalimentación instantánea para el atleta

### Captura de Retroalimentación
- Botón "Capturar" para guardar snapshots instantáneos
- Incluye: frame de cámara, postura, ángulos y feedback
- Descarga automática en formato PNG
- Almacenamiento en `backend/capturas/`

### Pantalla Completa
- Botón "Pantalla completa" para ampliar vista
- Atajo de teclado: **F**
- Ideal para entrenamientos en tiempo real

### Selector de Cámara
- Cambiar entre múltiples cámaras disponibles
- Persistencia de configuración

## 📖 Documentación del Backend

Para información sobre scripts de entrenamiento y estructura del backend, ver:
- [backend/README.md](backend/README.md)

## 🤖 Modelos de IA

### Vista Lateral
- Modelo: Árbol de decisión
- Features: Ángulos de brazos, rodilla y tronco
- Salida: CORRECTO o INCORRECTO

### Vista Frontal
- Modelo: Basado en simetría y alineación corporal
- Features: Simetría de extremidades, alineación, ancho de base
- Validación: Inclinación del tronco (máx. 10°)

## 🛠️ Tecnologías Utilizadas

- **Backend**: Flask (Python)
- **Frontend**: HTML5, CSS3, JavaScript vanilla
- **IA/ML**: MediaPipe, scikit-learn, pandas, numpy
- **Video**: OpenCV, getUserMedia API

## 📝 Notas Importantes

1. **Primera ejecución**: El sistema descargará automáticamente el modelo de MediaPipe (~200MB)
2. **Modelos entrenados**: Deben estar presentes en `backend/` :
   - `modelo_tecnica_antebrazos.pkl` (lateral)
   - `modelo_tecnica_antebrazos_frontal.pkl` (frontal)
3. **Capturas**: Se guardan en `backend/capturas/` y se descargan localmente
4. **Permisos**: El navegador solicitará permiso para acceder a la cámara

## 🔗 URLs Principales

- Inicio: `http://127.0.0.1:5000`
- Entrenamiento: `http://127.0.0.1:5000/biomecanica.html`
- API (frames): `http://127.0.0.1:5000/api/evaluar_frame` (POST)
- API (guardar): `http://127.0.0.1:5000/api/guardar_captura` (POST)

## 🤝 Contribuidores

Proyecto desarrollado para la Universidad Mariana.

## 📄 Licencia

Proyecto de investigación académica.

---

**¿Problemas?** Revisa que:
- ✅ El entorno virtual esté activado
- ✅ Las dependencias estén instaladas (`pip install -r requirements.txt`)
- ✅ El puerto 5000 esté disponible
- ✅ La cámara esté conectada y con permisos concedidos
