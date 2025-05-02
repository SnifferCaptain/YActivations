import torch
import torch.nn as nn

class SFReLU(nn.Module):
    """
    soft maxout optimized FReLU
    """
    def __init__(self, channels:int, k:int, bn_affine:bool = True):
        super().__init__()
        if k % 2 == 0:
            k = k + 1
            print("k must be odd number, auto switch to k = %d", k)
        self.cv = nn.Conv2d(channels, channels, k, 1, (k - 1) // 2, bias=False, groups=channels)
        self.bn = nn.BatchNorm2d(channels, affine=bn_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.bn(self.cv(x))
        x = self.max(f, x)
        return x

    def max(self, x0, x1):
        # speed optimized soft maxout, 100x faster on gpu while deploying onnx
        x = x0 - x1
        x = nn.functional.silu(x) # maxout = relu(x)
        x = x + x1
        return x
