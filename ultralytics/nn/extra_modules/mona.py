import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x

class MonaOp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1, groups=in_features)  # 3×3
        self.conv2 = nn.Conv2d(in_features, in_features, kernel_size=(3, 5), padding=(1, 2), groups=in_features)  # 3×5 水平
        self.conv3 = nn.Conv2d(in_features, in_features, kernel_size=(5, 3), padding=(2, 1), groups=in_features)  # 5×3 垂直
        self.conv4 = nn.Conv2d(in_features, in_features, kernel_size=5, padding=2, groups=in_features)  # 5×5

        self.projector = nn.Conv2d(in_features, in_features, kernel_size=1, )

    def forward(self, x):
        identity = x
        conv1_x = self.conv1(x)
        conv2_x = self.conv2(x)
        conv3_x = self.conv3(x)
        conv4_x = self.conv4(x)

        x = (conv1_x + conv2_x + conv3_x + conv4_x) / 4.0 + identity

        identity = x

        x = self.projector(x)

        return identity + x

class Mona(nn.Module):
    def __init__(self,
                 in_dim):
        super().__init__()

        self.project1 = nn.Conv2d(in_dim, 64, 1)
        self.nonlinear = F.gelu
        self.project2 = nn.Conv2d(64, in_dim, 1)

        self.dropout = nn.Dropout(p=0.1)

        self.adapter_conv = MonaOp(64)

        self.norm = LayerNorm2d(in_dim)
        self.gamma = nn.Parameter(torch.ones(in_dim, 1, 1) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(in_dim, 1, 1))

    def forward(self, x, hw_shapes=None):
        identity = x

        x = self.norm(x) * self.gamma + x * self.gammax

        project1 = self.project1(x)

        project1 = self.adapter_conv(project1)

        nonlinear = self.nonlinear(project1)
        nonlinear = self.dropout(nonlinear)
        project2 = self.project2(nonlinear)

        return identity + project2