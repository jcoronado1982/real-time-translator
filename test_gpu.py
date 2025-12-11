import sys

import torch


def check_gpu() -> None:
    print(f"Python version: {sys.version}")
    print(f"Torch version: {torch.__version__}")

    if torch.cuda.is_available():
        print("✅ GPU DETECTADA:")
        print(f"   Nombre: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   cuDNN Version: {torch.backends.cudnn.version()}")

        try:
            x = torch.rand(5, 5).cuda()
            print("✅ Prueba de tensor en GPU exitosa.")
        except Exception as e:
            print(f"❌ Error al usar la GPU: {e}")
    else:
        print("❌ GPU NO DETECTADA (Se usará CPU)")


if __name__ == "__main__":
    check_gpu()


