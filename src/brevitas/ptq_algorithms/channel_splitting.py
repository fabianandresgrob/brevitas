import torch
import torch.nn as nn

from brevitas.graph.base import GraphTransform
from brevitas.graph.equalize import _extract_regions


def _channels_maxabs(layer, dim=1):
    if isinstance(layer, nn.Conv2d):
        # get maximum per output channel
        max_per_channel = torch.max(layer.weight.abs().flatten(1), dim=dim).values

    elif isinstance(layer, nn.Linear):
        # get maximum per output channel
        max_per_channel = layer.weight.abs().max(dim=dim).values

    # get indices
    channels = torch.argsort(max_per_channel, descending=True)

    return channels


class channel_splitting_mode:

    def __init__(self, model, split_ratio: float = 0.02, split_criterion: str = 'maxabs') -> None:
        self.model = model
        self.split_ratio = split_ratio
        self.graph_cs = ChannelSplitting(
            model, split_ratio=split_ratio, split_criterion=split_criterion)

    def __enter__(self) -> None:
        # first find the regions according to the criterion
        pass

    def __exit__(self) -> None:
        pass


class ChannelSplitting(GraphTransform):

    def __init__(self, model, split_ratio=0.02, split_criterion='maxabs'):
        super(ChannelSplitting, self).__init__()
        self.graph_model = model

        self.regions = _extract_regions(self.graph_model)

        self.split_ratio = split_ratio
        self.split_criterion = split_criterion
        if self.split_criterion == 'maxabs':
            self.split_fn = _channels_maxabs
