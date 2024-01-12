from inspect import signature

import torch

INPUT_NAMES = ['input', 'inp', 'query', 'x', 'args']


class ChannelSplitModule(torch.nn.Module):

    def __init__(self, layer, channels_to_duplicate) -> None:
        super().__init__()

        self.layer = layer
        self.channels_to_duplicate = channels_to_duplicate

    def forward(self, *args, **kwargs):
        # Convert args + kwargs + defaults into kwargs
        bound_arguments = signature(self.layer.forward).bind(*args, **kwargs)
        bound_arguments.apply_defaults()
        kwargs = bound_arguments.arguments

        possible_input_kwargs = INPUT_NAMES
        input_kwarg = [x for x in kwargs.keys() if x in possible_input_kwargs][0]
        x = kwargs[input_kwarg]
        if input_kwarg == 'args':
            x = x[0]
            kwargs = {'args': x}
        out = x

        self.channels_to_duplicate = self.channels_to_duplicate.to(out.device)

        channels = torch.index_select(out, dim=1, index=self.channels_to_duplicate)
        out = torch.cat([out, channels], dim=1)

        kwargs[input_kwarg] = out

        out = self.layer(*kwargs.values())
        return out
