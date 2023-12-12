import math
from typing import Dict, List, Set, Tuple, Union
import warnings

import torch
import torch.nn as nn

from brevitas.fx import GraphModule
from brevitas.graph.base import GraphTransform
from brevitas.graph.equalize import _extract_regions


def _channels_maxabs(module, splits_per_layer, split_input):
    # works for Conv2d and Linear
    dim = 1 - int(split_input)
    # dim 0 -> input channels max, dim 1 -> output channels max!
    if len(module.weight.data.shape) > 1:
        if isinstance(module, nn.Conv2d):
            if not split_input:
                # gets the max value for each output channel
                max_per_channel = module.weight.data.abs().flatten(1).max(1).values
                # check if max_per_channel has the same length as output channels
                assert len(max_per_channel) == module.weight.shape[0]
            else:
                # getting max value for each input channel
                max_per_channel = module.weight.data.abs().max(0).values.flatten(1).max(1).values
                # check if same length as input channels
                assert len(max_per_channel) == module.weight.shape[1]
        elif isinstance(module, nn.Linear):
            max_per_channel = module.weight.data.abs().max(dim=dim).values.flatten()
        channels = torch.argsort(max_per_channel, descending=True)
    else:
        # BN etc. don't have multiple dimensions, so just do argsort desc
        channels = torch.argsort(module.weight.data, descending=True)
    return channels[:splits_per_layer]


def _channels_to_split(
        sources: List[nn.Module],
        sinks: List[nn.Module],
        split_criterion: str,
        split_ratio: float,
        split_input: bool) -> Dict[nn.Module, List[torch.Tensor]]:
    modules = sinks if split_input else sources
    # the modules are all of the same shape so we can just take the first one
    num_channels = modules[0].weight.shape[int(split_input)]
    total_splits = int(math.ceil(split_ratio * num_channels))
    # each channel in the modules selects their portion of the total channels to split
    splits_per_layer = int(math.floor(total_splits / len(modules)))

    if splits_per_layer == 0:
        warnings.warn(f'No splits for {modules}, increasing split_ratio could help.')

    module_to_channels = {}
    if split_criterion == 'maxabs':
        for module in modules:
            module_to_channels[module] = _channels_maxabs(module, splits_per_layer, split_input)

    # return tensor with the indices to split
    channels_to_split = torch.cat(list(module_to_channels.values()))
    return torch.unique(channels_to_split)


def _split_single_channel(channel, grid_aware: bool, split_factor: float):
    if split_factor == 1:
        # duplicates the channel
        return channel, channel

    if grid_aware:
        slice1 = channel - 0.5
        slice2 = channel + 0.5
        return slice1 * split_factor, slice2 * split_factor
    else:
        return channel * split_factor, channel * split_factor


def _split_channels(
        module, channels_to_split, grid_aware=True, split_input=False, split_factor=0.5) -> None:
    """
    Splits the channels `channels_to_split` of the `weights`.
    `split_input` specifies whether to split Input or Output channels.
    Can also be used to duplicate a channel, just set split_factor to 1.
    Returns: None
    """
    weight = module.weight.data
    bias = module.bias.data if module.bias is not None else None

    for id in channels_to_split:
        if isinstance(module, torch.nn.Conv2d):
            # there are four dimensions: [OC, IC, k, k]
            if split_input:
                channel = weight[:, id:id + 1, :, :]
                slice1, slice2 = _split_single_channel(channel, grid_aware, split_factor)
                weight = torch.cat((weight[:, :id, :, :], slice1, slice2, weight[:, id + 1:, :, :]),
                                   dim=1)
                module.in_channels += 1
            else:
                channel = weight[id:id + 1, :, :, :]
                slice1, slice2 = _split_single_channel(channel, grid_aware, split_factor)
                weight = torch.cat((weight[:id, :, :, :], slice1, slice2, weight[id + 1:, :, :, :]),
                                   dim=0)
                module.out_channels += 1

        elif isinstance(module, torch.nn.Linear):
            # there are two dimensions: [OC, IC]
            if split_input:
                channel = weight[:, id:id + 1]
                slice1, slice2 = _split_single_channel(channel, grid_aware, split_factor)
                weight = torch.cat((weight[:, :id], slice1, slice2, weight[:, id + 1:]), dim=1)
                module.in_features += 1
            else:
                channel = weight[id:id + 1, :]
                slice1, slice2 = _split_single_channel(channel, grid_aware, split_factor)
                weight = torch.cat((weight[:id, :], slice1, slice2, weight[id + 1:, :]), dim=0)
                module.out_features += 1

        if bias is not None and not split_input:
            channel = bias[id:id + 1] * split_factor
            bias = torch.cat((bias[:id], channel, channel, bias[id + 1:]))

    module.weight.data = weight
    if bias is not None:
        module.bias.data = bias


def _split_channels_region(
        sources: List[nn.Module],
        sinks: List[nn.Module],
        channels_to_split: torch.tensor,
        split_input: bool,
        grid_aware: bool = False):
    # splitting output channels
    # concat all channels that are split so we can duplicate those in the input channels later
    if not split_input:
        for module in sources:
            _split_channels(module, channels_to_split, grid_aware=grid_aware)
        for module in sinks:
            # then duplicate the input_channels for all modules in the sink
            _split_channels(
                module, channels_to_split, grid_aware=False, split_factor=1, split_input=True)
    else:
        # input channels are split in half, output channels duplicated
        for module in sinks:
            _split_channels(module, channels_to_split, grid_aware=grid_aware, split_input=True)
        for module in sources:
            # duplicate out_channels for all modules in the source
            _split_channels(module, channels_to_split, grid_aware=False, split_factor=1)


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

    for i, region in enumerate(regions):

        # check if region is suitable for channel splitting
        sources = [name_to_module[n] for n in region.srcs]
        sinks = [name_to_module[n] for n in region.sinks]

        if _is_supported(sources, sinks):
            # get channels to split
            channels_to_split = _channels_to_split(
                sources=sources,
                sinks=sinks,
                split_criterion=split_criterion,
                split_ratio=split_ratio,
                split_input=split_input)
            # splitting/duplicating channels
            _split_channels_region(
                sources=sources,
                sinks=sinks,
                channels_to_split=channels_to_split,
                grid_aware=grid_aware,
                split_input=split_input)

    return model


class ChannelSplitting(GraphTransform):

    def __init__(
            self, split_ratio=0.02, split_criterion='maxabs', grid_aware=False, split_input=False):
        super(ChannelSplitting, self).__init__()

        self.grid_aware = grid_aware
        self.split_ratio = split_ratio
        self.split_criterion = split_criterion
        self.split_input = split_input

    def apply(
            self,
            model,
            return_regions: bool = False
    ) -> Union[Tuple[GraphModule, Set[Tuple[str]]], GraphModule]:
        regions = _extract_regions(model)
        if len(regions) > 0:
            self.graph_model = _split(
                model=model,
                regions=regions,
                split_ratio=self.split_ratio,
                split_criterion=self.split_criterion,
                grid_aware=self.grid_aware,
                split_input=self.split_input)
        if return_regions:
            return self.graph_model, regions
        else:
            return self.graph_model
