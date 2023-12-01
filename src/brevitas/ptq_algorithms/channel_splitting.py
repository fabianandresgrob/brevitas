import math
from typing import Dict, List, Set, Tuple, Union

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
            # TODO make sure when splitting input channel this will return input channels
            max_per_channel = module.weight.abs().flatten(dim).max(dim).values
        elif isinstance(module, nn.Linear):
            max_per_channel = module.weight.data.abs().max(dim=dim).values.flatten()
        channels = torch.argsort(max_per_channel, descending=True)
    else:
        # BN etc. don't have multiple dimensions, so just do argsort desc
        channels = torch.argsort(module.weight.data, descending=True)
    return channels[:splits_per_layer]


def _channels_to_split(
        modules: List[nn.Module], split_criterion: str, split_ratio: float,
        split_input: bool) -> Dict[nn.Module, List[torch.Tensor]]:
    # the modules are all of the same shape so we can just take the first one
    num_channels = modules[0].weight.shape[int(split_input)]
    total_splits = int(math.ceil(split_ratio * num_channels))
    # each channel in the sources only splits a portion of the total splits
    if not split_input:
        splits_per_layer = int(math.floor(total_splits / len(modules)))
    else:
        # if we split input channels, each module has to split the whole budget
        splits_per_layer = total_splits
    assert splits_per_layer > 0, f"No channels to split in {modules} with split_rati {split_ratio}!"

    module_to_channels = {}
    if split_criterion == 'maxabs':
        for module in modules:
            module_to_channels[module] = _channels_maxabs(module, splits_per_layer, split_input)

    # return dict with modules as key and channels to split as value
    return module_to_channels


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
    bias = layer.bias.data

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


def _split_channels_region(
        module_to_split: Dict[nn.Module, torch.tensor],
        modules_to_duplicate: [nn.Module],
        split_input: bool,
        grid_aware: bool = False):
    # we are getting a dict[Module, channels to split]
    # splitting output channels
    # concat all channels that are split so we can duplicate those in the input channels later
    if not split_input:
        input_channels = torch.cat(list(module_to_split.values()))
        for module, channels in module_to_split.items():
            _split_channels(module, channels, grid_aware=grid_aware)
        for module in modules_to_duplicate:
            # then duplicate the input_channels for all modules in the sink
            _split_channels(
                module, input_channels, grid_aware=False, split_factor=1, split_input=True)
    else:
        # what if we split input channels of the sinks, which channels of the OC srcs have to duplicated?
        pass


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
                # we will have
                mod_to_channels = _channels_to_split(
                    sinks, split_criterion, split_ratio, split_input)
            else:
                mod_to_channels = _channels_to_split(srcs, split_criterion, split_ratio, False)
                _split_channels_region(
                    module_to_split=mod_to_channels,
                    modules_to_duplicate=sinks,
                    split_input=split_input)

            # now splits those channels that we just selected!

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
