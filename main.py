import torch

if __name__ == '__main__':
    print("CUDA availability:", torch.cuda.is_available())
    print("GPUs available:", torch.cuda.device_count())

