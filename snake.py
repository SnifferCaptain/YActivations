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