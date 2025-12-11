import torch
import transformers


def main() -> None:
    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")


if __name__ == "__main__":
    main()


