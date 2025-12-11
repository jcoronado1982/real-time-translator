import sys
import time
from pathlib import Path

from local_translator.src.stt import WhisperSTT
from local_translator.src.translation import NMTTranslator
from local_translator.src.tts import PiperTTS


def main() -> None:
    # Usamos el archivo de prueba que ya grabé
    audio_file = "prueba.wav"

    if not Path(audio_file).exists():
        print(f"❌ Error: No encuentro '{audio_file}'")
        return

    print(f"--- 🚀 Test Pipeline: {audio_file} ---")

    # Inicializar modelos (Forzamos CPU para evitar errores de memoria en la prueba)
    print("1. Cargando Whisper...")
    stt = WhisperSTT(model_size="small", device="cpu", compute_type="int8")

    print("2. Cargando Traductor...")
    translator = NMTTranslator(device="cpu")
    print("3. Cargando TTS (Piper)...")
    tts = PiperTTS()

    # Transcribir
    print("\n🎤 Transcribiendo...")
    t1 = time.time()
    text_es = stt.transcribe(audio_file)
    print(f"📝 Español: {text_es}")
    print(f"⏱️ Tiempo STT: {time.time() - t1:.2f}s")

    if text_es:
        # Traducir
        print("\n🇺🇸 Traduciendo...")
        t2 = time.time()
        text_en = translator.translate(text_es)
        print(f"🇺🇸 Inglés: {text_en}")
        print(f"⏱️ Tiempo MT: {time.time() - t2:.2f}s")

        if text_en:
            print("\n🔈 Reproduciendo con Piper...")
            tts.speak(text_en)
    else:
        print("⚠️ No se detectó texto en la transcripción; omitiendo traducción.")


if __name__ == "__main__":
    main()


