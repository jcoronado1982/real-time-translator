from __future__ import annotations

import os
import sys

# ==========================================
# 🔧 AUTO-CONFIGURACIÓN DE GPU (MÁGICO)
# ==========================================
def setup_gpu_environment():
    """
    Busca los drivers de NVIDIA dentro de Python y configura el sistema
    automáticamente para que no tengas que usar 'export LD_LIBRARY_PATH'.
    """
    # Si ya lo configuramos, no hacemos nada para evitar bucles
    if os.environ.get("TRANSLATOR_GPU_READY") == "1":
        return

    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
    except ImportError:
        print("⚠️ No se encontraron librerías NVIDIA pip. Usando sistema base.")
        return

    # Buscamos dónde están escondidos los archivos de la GPU
    cublas_path = os.path.dirname(nvidia.cublas.lib.__file__)
    cudnn_path = os.path.dirname(nvidia.cudnn.lib.__file__)
    
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    
    # Si no están en la ruta, los metemos y reiniciamos el programa
    if cublas_path not in current_ld:
        print("🔧 [Sistema] Inyectando drivers NVIDIA y reiniciando...")
        new_ld = f"{current_ld}:{cublas_path}:{cudnn_path}" if current_ld else f"{cublas_path}:{cudnn_path}"
        
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = new_ld
        env["TRANSLATOR_GPU_READY"] = "1" # Marca de que ya está listo
        
        # Reinicia el script con los nuevos superpoderes
        os.execve(sys.executable, [sys.executable] + sys.argv, env)

# Ejecutamos esto ANTES de importar torch
setup_gpu_environment()

# ==========================================
# 🚀 TU CÓDIGO ORIGINAL OPTIMIZADO
# ==========================================

import tempfile
import time
import speech_recognition as sr
import torch
import numpy as np

# Importamos tus módulos
from local_translator.src.stt import WhisperSTT
from local_translator.src.translation.helsinki_translator import HelsinkiTranslator
from local_translator.src.tts import PiperTTS

# --- FUNCIÓN: MATA-BUCLES ---
def is_looping(text: str) -> bool:
    """
    Detecta si Whisper entró en un bucle infinito (ej: "ya se ve, ya se ve")
    """
    if len(text) < 50:
        return False 
        
    words = text.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2: 
            return True

    mid = len(text) // 2
    part1 = text[:mid]
    part2 = text[mid:]
    
    if part1 in part2 or part2 in part1:
         return True
         
    return False

# --- FUNCIÓN: PORTERO IA (VAD) ---
def check_human_voice(wav_path: str, model, utils) -> bool:
    (get_speech_timestamps, _, read_audio, _, _) = utils
    wav = read_audio(wav_path, sampling_rate=16000)
    
    # Umbral 0.6 para ser estricto con el ruido de fondo
    speech_timestamps = get_speech_timestamps(wav, model, sampling_rate=16000, threshold=0.6)
    return len(speech_timestamps) > 0

def main() -> None:
    print("🛡️  INICIANDO SISTEMA PRO V2 (GPU Auto-Config + Anti-Bucles)...")

    print("   -> Cargando Silero VAD...")
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, trust_repo=True)
    
    print("   -> Cargando Motores IA...")
    # TU CONFIGURACIÓN FAVORITA: Base + Int8 (La más rápida)
    stt = WhisperSTT(model_size="base", device="cuda", compute_type="int8")
    translator = HelsinkiTranslator(device="cuda")
    tts = PiperTTS()

    recognizer = sr.Recognizer()
    
    # CONFIGURACIÓN FINA (Tus valores)
    recognizer.energy_threshold = 300  
    recognizer.dynamic_energy_threshold = False 
    
    # TIEMPOS (Tus valores para baja latencia)
    recognizer.pause_threshold = 0.7       
    recognizer.non_speaking_duration = 0.2
    
    # LISTA NEGRA
    forbidden_phrases = [
        "subscribe", "suscríbete", "subtítulos", "copyright", 
        "moo", "you", "thank you", "gracias por ver", "mbc"
    ]

    try:
        with sr.Microphone() as source:
            print("\n🎧 Calibrando silencio (1s)...")
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
            # Forzamos mínimo 300 para tu micro M-Audio
            if recognizer.energy_threshold < 300:
                recognizer.energy_threshold = 300
            
            print(f"   -> Sensibilidad: {recognizer.energy_threshold}")
            print("\n✅ LISTO. Habla.")
            
            while True:
                try:
                    print("\n🎤 Escuchando...")
                    audio = recognizer.listen(source, timeout=None, phrase_time_limit=None)

                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                        tmp.write(audio.get_wav_data())
                        tmp_path = tmp.name

                    try:
                        # 1. CHECK VAD (¿Es humano?)
                        is_human = check_human_voice(tmp_path, vad_model, vad_utils)
                        if not is_human:
                            print("   🗑️ Ruido detectado.")
                            continue 
                        
                        # 2. TRANSCRIPCIÓN
                        t0 = time.time()
                        text_es = stt.transcribe(tmp_path)
                        
                        if not text_es or len(text_es.strip()) < 2:
                            continue

                        # 3. DETECCIÓN DE BUCLES
                        if is_looping(text_es):
                            print(f"   🔄 BUCLE DETECTADO Y ELIMINADO: '{text_es[:30]}...'")
                            continue

                        # 4. LIMPIEZA DE ALUCINACIONES
                        clean = text_es.strip().lower()
                        if any(p in clean for p in forbidden_phrases):
                            print(f"   ⚠️ Alucinación bloqueada: '{text_es}'")
                            continue

                        dt = time.time() - t0
                        print(f"📝 ES: {text_es}  (⏱️ {dt:.2f}s)")

                        # 5. TRADUCCIÓN Y VOZ
                        text_en = translator.translate(text_es)
                        print(f"🇺🇸 EN: {text_en}")

                        if text_en:
                            tts.speak(text_en)

                    finally:
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print(f"⚠️ {e}")

    except KeyboardInterrupt:
        print("\n👋 Fin.")

if __name__ == "__main__":
    main()