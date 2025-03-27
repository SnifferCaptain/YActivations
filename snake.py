class SnakeA(nn.Module):
    """ SnakeA: y=tanh(x) + relu(x) """
    def __init__(self):
        super(SnakeA, self).__init__()

    def forward(self, x):
        return torch.tanh(x) + torch.relu(x)

class SnakeB(nn.Module):
    """ SnakeB: y=tanh(x) + silu(x) """
    def __init__(self):
        super(SnakeB, self).__init__()

    def forward(self, x):
        return torch.tanh(x) + nn.functional.silu(x)

class SnakeC(nn.Module):
    """ SnakeB: y=tanh(x) + silu(x) """
    def __init__(self):
        super(SnakeC, self).__init__()

    def forward(self, x):
        return torch.erf(x) + nn.functional.gelu(x, 'tanh')

class DytSnakeB(nn.Module):
    """ DytSnakeB: y=tanh(αx) + silu(x) """
    def __init__(self, shape:list[int]=[1]):
        super(DytSnakeB, self).__init__()
        self.alpha = nn.Parameter(torch.randn(shape)*0.1+1, requires_grad=True)
        self.register_full_backward_hook(self.hook)

    def forward(self, x):
        return torch.tanh(self.alpha * x) + nn.functional.silu(x)

    def hook(self, module, grad_input, grad_output):
        self.alpha.grad = self.alpha.grad * 0.1
