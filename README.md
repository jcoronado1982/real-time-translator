# Local Real-Time Translator

Sistema de traducción de voz en tiempo real ejecutándose localmente. Convierte voz en español a voz en inglés con baja latencia.

## 💻 Requisitos del Sistema

- **Sistema Operativo**: Linux (Ubuntu 22.04+ recomendado).
- **GPU**: Tarjeta gráfica NVIDIA (GTX 1060 o superior recomendado) con drivers instalados.
- **Audio**: Micrófono funcional.
- **Software**: Python 3.10+.

## ⚙️ Instalación

### 1. Dependencias del Sistema
Instala las herramientas necesarias para audio y procesamiento:

```bash
sudo apt-get update
sudo apt-get install python3-pip python3-venv portaudio19-dev libasound2-dev ffmpeg espeak-ng
```

### 2. Dependencias de Python
Instala las librerías del proyecto:

```bash
pip install -r requirements.txt
```

## 🚀 Cómo Arrancar

Simplemente ejecuta el script principal con VAD (Voice Activity Detection):

```bash
python3 live_translator_vad.py
```

> **Nota**: No es necesario exportar `LD_LIBRARY_PATH` ni configurar variables de entorno manualmente. El script detecta tus drivers NVIDIA y se autoconfigura al iniciar.

## 🎛️ Guía de Configuración (Tuning)

Puedes ajustar el comportamiento del traductor editando las variables al inicio de `live_translator_vad.py`:

| Variable | Valor Recomendado | Descripción |
| :--- | :--- | :--- |
| `recognizer.pause_threshold` | `0.6` - `0.8` | **Paciencia**. Tiempo (segundos) de silencio para considerar que una frase terminó. Valores más bajos = más rapidez pero corta frases. |
| `recognizer.energy_threshold` | `300` - `500` | **Sensibilidad**. Nivel mínimo de volumen para activar la escucha. Si hay mucho ruido ambiente, sube este valor. |
| `model_size` | `"base"` | **Velocidad vs Precisión**. Usa `"base"` para máxima velocidad. Usa `"small"` si necesitas más precisión en la transcripción. |

## ❓ Solución de Problemas

### "No me escucha / No hace nada"
1. Asegúrate de que tu micrófono está seleccionado como dispositivo de entrada predeterminado en la configuración de **Sonido de Ubuntu**.
2. Ejecuta la herramienta de diagnóstico:
   ```bash
   python3 check_system.py
   ```
   Esto intentará reparar automáticamente la selección del dispositivo de audio.

### "Repite frases constantemente"
Esto es una "alucinación" común en modelos de IA cuando hay silencio o ruido estático.
- El sistema incluye un **filtro Anti-Bucle** que bloquea la mayoría.
- Si persiste, intenta subir el `recognizer.energy_threshold` o alejar el micrófono de fuentes de ruido (ventiladores, etc.).
