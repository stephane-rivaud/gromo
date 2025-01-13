import torch


if __name__ == "__main__":
    print(f"torch.__version__: {torch.__version__}")
    print(f"torch.version.cuda: {torch.version.cuda}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")
    print(f"torch.cuda.get_device_capability(): {torch.cuda.get_device_capability()}")
