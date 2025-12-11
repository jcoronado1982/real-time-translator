from __future__ import annotations

import time
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from local_translator.src.utils.logger import get_logger


class NMTTranslator:
    """
    Spanish -> English translation using Helsinki-NLP/opus-mt-es-en.
    Loads model/tokenizer once to avoid per-call overhead.
    """

    def __init__(
        self,
        model_name: str = "Helsinki-NLP/opus-mt-es-en",
        device: Optional[str] = None,
    ) -> None:
        self._log = get_logger(__name__)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self._log.info("Translator loaded on %s", self.device)

    def translate(self, text: str) -> str:
        """
        Translate Spanish text to English.
        """
        if not text:
            return ""

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        with torch.no_grad():
            generated_tokens = self.model.generate(**encoded, max_length=256)

        output_text = self.tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )
        return output_text[0] if output_text else ""


if __name__ == "__main__":
    sample = "Hola, soy un desarrollador de software con 10 años de experiencia."
    translator = NMTTranslator()
    start = time.time()
    result = translator.translate(sample)
    elapsed = time.time() - start
    print(f"Input: {sample}")
    print(f"Output: {result}")
    print(f"Inference time: {elapsed:.3f}s")


