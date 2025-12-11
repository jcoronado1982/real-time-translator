from __future__ import annotations

import time
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from local_translator.src.utils.logger import get_logger


class HelsinkiTranslator:
    """
    Spanish -> English translation using Helsinki-NLP/opus-mt-es-en.
    Loads model/tokenizer once to avoid per-call overhead.
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-es-en",
        model_dir: Optional[str] = None,
        device: Optional[str] = "cuda",
    ) -> None:
        self._log = get_logger(__name__)

        # Validación de dispositivo
        if device == "cuda" and not torch.cuda.is_available():
            self._log.warning("CUDA no disponible; volviendo a CPU.")
            device = "cpu"
        
        self.device = torch.device(device)
        self._log.info(f"Inicializando traductor en: {self.device}")

        # Si nos pasan un directorio, lo usamos como cache_dir para guardar los modelos allí
        cache_dir = model_dir if model_dir else None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=cache_dir)
        except Exception as e:
            self._log.error(f"Error cargando el modelo {model_name}: {e}")
            raise e

        # Mover modelo a GPU/CPU
        self.model.to(self.device)

        # Optimización FP16 si estamos en CUDA
        if self.device.type == "cuda":
            self.model.half()

        self.model.eval()
        self._log.info(f"Translator loaded successfully on {self.device}")

    def translate(self, text: str) -> str:
        """
        Translate Spanish text to English.
        """
        if not text or not text.strip():
            return ""

        try:
            # Tokenizar
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self.device)

            # Generar traducción
            with torch.no_grad():
                generated_tokens = self.model.generate(
                    **encoded, 
                    max_length=256,
                    num_beams=4,        # Mejora un poco la calidad
                    early_stopping=True
                )

            # Decodificar
            output_text = self.tokenizer.batch_decode(
                generated_tokens, skip_special_tokens=True
            )
            return output_text[0] if output_text else ""
            
        except Exception as e:
            self._log.error(f"Error durante traducción: {e}")
            return ""


if __name__ == "__main__":
    # Prueba rápida
    sample = "Hola, soy un desarrollador de software con 10 años de experiencia."
    
    print("Cargando modelo...")
    translator = HelsinkiTranslator()
    
    start = time.time()
    result = translator.translate(sample)
    elapsed = time.time() - start
    
    print(f"Input: {sample}")
    print(f"Output: {result}")
    print(f"Inference time: {elapsed:.3f}s")