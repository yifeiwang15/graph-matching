import torch

class Node:
    def __init__(self, data):
        self.w = torch.rand(data.shape)
        self.attr = torch.tensor(data)

