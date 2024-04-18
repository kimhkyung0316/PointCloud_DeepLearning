import torch


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    return dist