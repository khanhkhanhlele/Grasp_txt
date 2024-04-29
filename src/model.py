import torch
import torch.nn as nn

class GraspLLava(nn.Module):
    def __init__(self):
        super(GraspLLava, self).__init__()
        self.llava = nn.Linear(10, 2)