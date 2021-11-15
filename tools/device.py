import torch

if __name__ == "__main__":
    print("GPU" if torch.cuda.is_available() else "CPU")
