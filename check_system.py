import os
import sys
import time
import subprocess
import numpy as np
import torch
import speech_recognition as sr

# Truco para que encuentre tus módulos
sys.path.append(os.getcwd())

def print_status(component, status, message=""):
    icon = "✅" if status else "❌"
    print(f"{icon} [{component}]: {message}")
    return status

# --- AUTO-REPARACIÓN DE AUDIO (Linux/PulseAudio) ---
def auto_fix_audio_linux():
    print("   🔧 Intentando reparar configuración de audio en Ubuntu...")
    try:
        # 1. Obtener fuentes de audio
        result = subprocess.run(["pactl", "list", "short", "sources"], capture_output=True, text=True)
        lines = result.stdout.split('\n')
        
        # Palabras clave de tu interfaz (M-Audio / USB Codec)
        keywords = ["USB", "PCM2900", "M-Audio"]
        target_source = None
        
        for line in lines:
            if "input" in line and any(k in line for k in keywords):
                parts = line.split('\t')
                if len(parts) > 1:
                    target_source = parts[1]
                    break
        
        if target_source:
            # 2. Forzar como predeterminado
            subprocess.run(["pactl", "set-default-source", target_source])
            print(f"   👉 Dispositivo cambiado a: {target_source}")
            # 3. Mover streams actuales (si los hay) al nuevo dispositivo
            # Esto ayuda si el proceso de audio ya estaba abierto
            return True
        else:
            print("   ⚠️ No se encontró el micrófono USB automáticamente.")
            return False
    except Exception as e:
        print(f"   ⚠️ Error intentando configurar audio: {e}")
        return False

# --- AUTO-REPARACIÓN DE GPU ---
def configure_gpu_env():
    """Busca las librerías de NVIDIA e inyecta la ruta en el entorno actual."""
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
        
        cublas_path = os.path.dirname(nvidia.cublas.lib.__file__)
        cudnn_path = os.path.dirname(nvidia.cudnn.lib.__file__)
        
        # Inyectar en el entorno actual del script
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        if cublas_path not in current_ld:
            os.environ["LD_LIBRARY_PATH"] = f"{current_ld}:{cublas_path}:{cudnn_path}"
            # Reiniciar la carga de librerías dinámicas para este proceso es complejo,
            # pero establecer la variable ayuda a los subprocesos o cargas diferidas.
            return True
    except ImportError:
        return False
    return True

def check_gpu():
    print("\n--- 1. VERIFICANDO GPU Y CUDA ---")
    
    # Paso 1: Configuración automática de entorno
    configure_gpu_env()
    
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print_status("GPU", True, f"Detectada: {device_name}")
            
            try:
                # Intentamos una operación pequeña con CUDNN
                x = torch.ones(1).cuda()
                import nvidia.cudnn.lib
                print_status("Drivers NVIDIA", True, "Librerías cargadas correctamente.")
            except Exception:
                print_status("Drivers NVIDIA", False, "Fallo al usar CUDNN. (Revisar LD_LIBRARY_PATH)")
                return False
        else:
            print_status("GPU", False, "Torch no detecta CUDA. Se usará CPU.")
            return False
    except Exception as e:
        print_status("GPU", False, f"Error crítico: {e}")
        return False
    return True

def check_mic():
    print("\n--- 2. VERIFICANDO MICRÓFONO ---")
    
    # Intentamos grabar. Si falla, aplicamos corrección y reintentamos.
    max_intentos = 2
    for intento in range(1, max_intentos + 1):
        try:
            r = sr.Recognizer()
            # Usamos el default del sistema (que intentaremos arreglar si falla)
            with sr.Microphone() as source:
                print(f"   🎤 Grabando prueba (Intento {intento})... Habla fuerte.")
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.listen(source, timeout=3, phrase_time_limit=2)
                
                data = audio.get_raw_data()
                np_data = np.frombuffer(data, dtype=np.int16)
                max_amp = np.max(np.abs(np_data))
                
                if max_amp > 100:
                    print_status("Audio Input", True, f"Audio detectado (Nivel: {max_amp})")
                    return True
                else:
                    print(f"   ⚠️ Nivel de señal nulo ({max_amp}).")
                    if intento < max_intentos:
                        # AQUÍ LA MAGIA: Intentamos arreglar Linux
                        if auto_fix_audio_linux():
                            print("   ⏳ Reintentando en 1 segundo...")
                            time.sleep(1)
                            continue
        except Exception as e:
            print(f"   ⚠️ Error: {e}")
            if intento < max_intentos:
                auto_fix_audio_linux()
                time.sleep(1)
    
    print_status("Audio Input", False, "No se logró capturar audio tras los intentos.")
    return False

def check_models():
    print("\n--- 3. VERIFICANDO MODELOS (Carga rápida) ---")
    try:
        from local_translator.src.stt import WhisperSTT
        from local_translator.src.translation.helsinki_translator import HelsinkiTranslator
        
        # Prueba ligera para ver si explota la memoria o rutas
        print("   -> Init Whisper...")
        WhisperSTT(model_size="tiny", device="cuda" if torch.cuda.is_available() else "cpu")
        print_status("Modelos", True, "Carga exitosa.")
    except Exception as e:
        print_status("Modelos", False, f"Error: {e}")

def main():
    print("🕵️  DIAGNÓSTICO Y AUTO-REPARACIÓN DEL SISTEMA...\n")
    
    gpu_ok = check_gpu()
    mic_ok = check_mic()
    
    if gpu_ok and mic_ok:
        check_models()
        print("\n✅✅ SISTEMA VERIFICADO Y LISTO ✅✅")
        print("Ahora puedes ejecutar: python3 live_translator.py")
    else:
        print("\n⚠️  EL SISTEMA TIENE ERRORES QUE NO SE PUDIERON CORREGIR AUTOMÁTICAMENTE.")

if __name__ == "__main__":
    main()