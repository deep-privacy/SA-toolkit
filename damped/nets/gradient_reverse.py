import torch

class GradientReverse(torch.autograd.Function):
    """
    Identical mapping from input to output
    but reverse the gradient during backwards
    """
    scale = 0.1
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return GradientReverse.scale * grad_output.neg()
    

def grad_reverse(x, scale=0.1):
    GradientReverse.scale = scale
    return GradientReverse.apply(x)
