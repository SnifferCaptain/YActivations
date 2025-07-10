import torch
import torch.nn as nn

def softResial(x:torch.Tensor, y:torch.Tensor):
    """
    the function will behave like this:
    when x + y >> 0, it will be x + y
    when x + y << 0, it will be min(x, y)
    when x + y ~~ 0, it will be a switching between x+y and min(x, y)
    """
    sub = x - y
    add = x + y
    x = x - nn.functional.silu(sub) + nn.functional.silu(add)
    return x
