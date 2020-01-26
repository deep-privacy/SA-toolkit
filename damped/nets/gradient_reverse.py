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


def grad_reverse_net(net: type):
    """
    create a new module where the first Function applied to the input feature
    of the forward function is ``grad_reverse``.

    This function doesn't changes how the nn.Module is serialized/unserialized.

    Args:
        net (type[torch.nn.Module]): The class name (not the initialized) object.
    """

    class GradientReverseProxy(net):
        """
        Helper class that apply grad_reverse at in the input feat, then pass to the
        embedded.
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__class__.__name__ = "GradientReverse-" + net.__name__
            self.scale = 10
            print("Gradient reversed!")

        def forward(self, hs_pad):
            GradientReverse.scale = self.scale
            x = GradientReverse.apply(hs_pad)
            return super().forward(x)

    return GradientReverseProxy
