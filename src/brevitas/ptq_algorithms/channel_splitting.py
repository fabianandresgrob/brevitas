import math
from typing import Dict, List, Set, Tuple, Union

import torch
import torch.nn as nn

from brevitas.fx import GraphModule
from brevitas.graph.base import GraphTransform
from brevitas.graph.equalize import _extract_regions


def _channels_maxabs(module, split_input):
    # works for Conv2d and Linear
    dim = 1 - int(split_input)
    # dim 0 -> input channels max, dim 1 -> output channels max!
    max_per_channel = module.weight.data.abs().max(dim=dim).values.flatten()
    channels = torch.argsort(max_per_channel, descending=True)
    return channels


def _channels_to_split(
        modules: List[nn.Module], split_criterion: str, split_ratio: float,
        split_input: bool) -> List[int]:
    # how are we taking channels across multiple sources?
    # the sources must all have the same num of output channels, if we split output channels
    # the criterion could also give different channels to split for each source, so we have to think
    # about how to select a intersection of channels to split for each source

    num_channels = module.weight.shape[int(split_input)]
    num_channels_to_split = int(math.ceil(split_ratio * num_channels))

    module_to_channels = {}
    if split_criterion == 'maxabs':
        for module in modules:
            channels = _channels_maxabs(module, split_input)
            module_to_channels[module] = channels

    # select the intersection of channels as the channels to split
    # Note: this is just one approach of selecting a shared subset of channels to choose, this might end up not selecting any channel or using less channels than
    channels_to_split = set.intersection(*map(set, module_to_channels.values()))

    return channels_to_split[:num_channels_to_split]


def _split_channels(
        layer, channels_to_split, grid_aware=True, split_input=False, split_factor=0.5) -> None:
    """
    Splits the channels `channels_to_split` of the `weights`.
    `split_input` specifies whether to split Input or Output channels.
    Can also be used to duplicate a channel, just set split_factor to 1.
    Returns: None
    """
    # change it to .data attribute
    weight = layer.weight.data

    for id in channels_to_split:
        if isinstance(layer, torch.nn.Conv2d):
            # there are four dimensions: [OC, IC, k, k]
            if split_input:
                channel = weight[:, id:id + 1, :, :] * split_factor
                weight = torch.cat(
                    (weight[:, :id, :, :], channel, channel, weight[:, id + 1:, :, :]), dim=1)
                layer.in_channels += 1
            else:
                # split output
                channel = weight[id:id + 1, :, :, :] * split_factor
                # duplicate channel
                weight = torch.cat(
                    (weight[:id, :, :, :], channel, channel, weight[id + 1:, :, :, :]), dim=0)
                layer.out_channels += 1

        elif isinstance(layer, torch.nn.Linear):
            # there are two dimensions: [OC, IC]
            if split_input:
                # simply duplicate channel
                channel = weight[:, id:id + 1] * split_factor
                weight = torch.cat((weight[:, :id], channel, channel, weight[:, id + 1:]), dim=1)
                layer.in_features += 1
            else:
                # split output
                channel = weight[id:id + 1, :] * split_factor
                weight = torch.cat((weight[:id, :], channel, channel, weight[id + 1:, :]), dim=0)
                layer.out_features += 1

        if bias is not None and not split_input:
            # also split bias
            channel = layer.bias.data[id:id + 1] * split_factor
            bias = torch.cat((bias[:id], channel, channel, bias[id + 1:]))
            layer.bias.data = bias

    # setting the weights as the new data
    layer.weight.data = weight


def _split_channels_region(srcs, sinks, channels, split_ratio, grid_aware, split_input):
    if split_input:
        for module in srcs:
            _split_channels(
                module,
                channels,
                split_ratio,
                grid_aware=grid_aware,
                split_input=True,
                split_factor=1)
        for module in sinks:
            _split_channels(
                module,
                channels,
                split_ratio,
                grid_aware=grid_aware,
                split_input=True,
                split_factor=0.5)
    else:
        for module in srcs:
            _split_channels(
                module,
                channels,
                split_ratio,
                grid_aware=grid_aware,
                split_input=False,
                split_factor=0.5)
        for module in sinks:
            _split_channels(
                module,
                channels,
                split_ratio,
                grid_aware=grid_aware,
                split_input=False,
                split_factor=1)


def _is_supported(srcs: List[nn.Module], sinks: List[nn.Module]) -> bool:
    # check if OCs of sources are all equal
    srcs_ocs = set(module.weight.shape[0] for module in srcs)
    if len(srcs_ocs) > 1:
        return False

    # check if ICs of sinks are all equal
    sinks_ics = set(module.weight.shape[1] for module in sinks)
    if len(sinks_ics) > 1:
        return False

    return srcs_ocs == sinks_ics


def _split(
        model: GraphModule,
        regions: Set[Tuple[str]],
        split_ratio: float,
        split_criterion: str,
        grid_aware: bool,
        split_input: bool) -> GraphModule:
    name_to_module: Dict[str, nn.Module] = {}
    name_set = set()
    for region in regions:
        for name in region.srcs:
            name_set.add(name)
        for name in region.sinks:
            name_set.add(name)

    for name, module in model.named_modules():
        if name in name_set:
            name_to_module[name] = module

    for region in regions:

        # check if region is suitable for channel splitting
        srcs = [name_to_module[n] for n in region.srcs]
        sinks = [name_to_module[n] for n in region.sinks]

        if _is_supported(srcs, sinks):

            # get channels to split
            if split_input:
                channels = _channels_to_split(sinks, split_criterion, split_ratio, True)
            else:
                channels = _channels_to_split(srcs, split_criterion, split_ratio, False)

            # split channels across regions
            _split_channels_region(
                srcs=srcs,
                sinks=sinks,
                channels=channels,
                split_ratio=split_ratio,
                grid_aware=grid_aware)

    return model


class ChannelSplitting(GraphTransform):

    def __init__(
            self,
            model,
            split_ratio=0.02,
            split_criterion='maxabs',
            grid_aware=False,
            split_input=False):
        super(ChannelSplitting, self).__init__()
        self.graph_model = model
        self.grid_aware = grid_aware

        self.split_ratio = split_ratio
        self.split_criterion = split_criterion
        self.split_input = split_input

    def apply(
            self,
            return_regions: bool = False
    ) -> Union[Tuple[GraphModule, Set[Tuple[str]]], GraphModule]:
        regions = _extract_regions(self.graph_model)
        if len(regions) > 0:
            self.graph_model = _split(
                model=self.graph_model,
                regions=regions,
                split_ratio=self.split_ratio,
                split_criterion=self.split_criterion,
                grid_aware=self.grid_aware,
                split_input=self.split_input)
        if return_regions:
            return self.graph_model, regions
        else:
            return self.graph_model
