import math
from typing import Dict, List, Set, Tuple, Union
import warnings

import torch
import torch.nn as nn

from brevitas.fx import GraphModule
from brevitas.fx import Node
from brevitas.graph.base import GraphTransform
from brevitas.graph.base import ModuleInstanceToModuleInstance
from brevitas.graph.equalize import _extract_regions
from brevitas.graph.utils import get_module
from brevitas.nn import ChannelSplitModule


def _calculate_scale(weights, bit_width, clip_threshold=1.):
    max_abs = weights.abs().max()
    clip_max_abs = max_abs * clip_threshold
    n = 2 ** (bit_width - 1) - 1
    return n / clip_max_abs


def _channels_maxabs(module, splits_per_layer, split_input):
    # works for Conv2d and Linear
    dim = 1 - int(split_input)
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
    return channels[:splits_per_layer]


def _channels_to_split(
        sources: Dict[str, nn.Module],
        sinks: Dict[str, nn.Module],
        split_criterion: str,
        split_ratio: float,
        split_input: bool) -> Dict[nn.Module, List[torch.Tensor]]:
    modules = sinks if split_input else sources
    # the modules are all of the same shape so we can just take the first one
    num_channels = next(iter(modules.values())).weight.shape[int(split_input)]
    splits_per_layer = int(math.ceil(split_ratio * num_channels))

    if splits_per_layer == 0:
        warnings.warn(f'No splits for {modules}, increasing split_ratio could help.')

    module_to_channels = {}
    if split_criterion == 'maxabs':
        for name, module in modules.items():
            module_to_channels[name] = _channels_maxabs(module, splits_per_layer, split_input)

    # return tensor with the indices to split
    channels_to_split = torch.cat(list(module_to_channels.values()))
    return torch.unique(channels_to_split)


def _split_single_channel(channel, grid_aware: bool, split_factor: float, scale: float = 1.):
    if grid_aware:
        split_channel = channel * split_factor * scale
        slice1 = (split_channel - 0.25) / scale
        slice2 = (split_channel + 0.25) / scale
        return slice1, slice2
    else:
        return channel * split_factor, channel * split_factor


def _split_channels(
        module,
        channels_to_split,
        grid_aware=True,
        split_input=False,
        split_factor=0.5,
        bit_width=8) -> None:
    """
    Splits the channels `channels_to_split` of the `weights`.
    `split_input` specifies whether to split Input or Output channels.
    Can also be used to duplicate a channel, just set split_factor to 1.
    Returns: None
    """
    weight = torch.clone(module.weight.data)
    bias = torch.clone(module.bias.data) if module.bias is not None else None

    # init scale
    scale = 1.

    if grid_aware:
        # do a preliminary split of the channels to get the scale for the split channels
        for id in channels_to_split:
            if isinstance(module, torch.nn.Conv2d):
                # there are four dimensions: [OC, IC, k, k]
                if split_input:
                    channel = weight[:, id:id + 1, :, :]
                    slice1, slice2 = _split_single_channel(channel=channel, grid_aware=False, split_factor=split_factor)
                    weight = torch.cat(
                        (weight[:, :id, :, :], slice1, slice2, weight[:, id + 1:, :, :]), dim=1)
                else:
                    channel = weight[id:id + 1, :, :, :]
                    slice1, slice2 = _split_single_channel(channel=channel, grid_aware=False, split_factor=split_factor)
                    weight = torch.cat(
                        (weight[:id, :, :, :], slice1, slice2, weight[id + 1:, :, :, :]), dim=0)

            elif isinstance(module, torch.nn.Linear):
                # there are two dimensions: [OC, IC]
                if split_input:
                    channel = weight[:, id:id + 1]
                    slice1, slice2 = _split_single_channel(channel=channel, grid_aware=False, split_factor=split_factor)
                    weight = torch.cat((weight[:, :id], slice1, slice2, weight[:, id + 1:]), dim=1)
                else:
                    channel = weight[id:id + 1, :]
                    slice1, slice2 = _split_single_channel(channel=channel, grid_aware=False, split_factor=split_factor)
                    weight = torch.cat((weight[:id, :], slice1, slice2, weight[id + 1:, :]), dim=0)
        # now calculate the scale of the split weights
        scale = _calculate_scale(weight, bit_width)

        # reset the weight variable
        weight = torch.clone(module.weight.data)

    for id in channels_to_split:
        if isinstance(module, torch.nn.Conv2d):
            # there are four dimensions: [OC, IC, k, k]
            if split_input:
                channel = weight[:, id:id + 1, :, :]
                slice1, slice2 = _split_single_channel(channel=channel, grid_aware=grid_aware, split_factor=split_factor, scale=scale)
                weight = torch.cat((weight[:, :id, :, :], slice1, weight[:, id + 1:, :, :], slice2),
                                   dim=1)
                module.in_channels += 1
            else:
                channel = weight[id:id + 1, :, :, :]
                slice1, slice2 = _split_single_channel(channel=channel, grid_aware=grid_aware, split_factor=split_factor, scale=scale)
                weight = torch.cat((weight[:id, :, :, :], slice1, weight[id + 1:, :, :, :], slice2),
                                   dim=0)
                module.out_channels += 1

        elif isinstance(module, torch.nn.Linear):
            # there are two dimensions: [OC, IC]
            if split_input:
                channel = weight[:, id:id + 1]
                slice1, slice2 = _split_single_channel(channel=channel, grid_aware=grid_aware, split_factor=split_factor, scale=scale)
                weight = torch.cat((weight[:, :id], slice1, weight[:, id + 1:], slice2), dim=1)
                module.in_features += 1
            else:
                channel = weight[id:id + 1, :]
                slice1, slice2 = _split_single_channel(channel=channel, grid_aware=grid_aware, split_factor=split_factor, scale=scale)
                weight = torch.cat((weight[:id, :], slice1, weight[id + 1:, :], slice2), dim=0)
                module.out_features += 1

        if bias is not None and not split_input:
            channel = bias[id:id + 1] * split_factor
            bias = torch.cat((bias[:id], channel, bias[id + 1:], channel))

    module.weight.data = weight
    if bias is not None:
        module.bias.data = bias


def _split_channels_region(
        sources: Dict[str, nn.Module],
        sinks: Dict[str, nn.Module],
        channels_to_split: torch.tensor,
        split_input: bool,
        grid_aware: bool = False,
        weight_bit_width: int = 8):
    # splitting output channels
    # concat all channels that are split so we can duplicate those in the input channels later
    if not split_input:
        for name, module in sources.items():
            _split_channels(
                module, channels_to_split, grid_aware=grid_aware, bit_width=weight_bit_width)
        for name, module in sinks.items():
            # then duplicate the input_channels for all modules in the sink
            _split_channels(
                module, channels_to_split, grid_aware=False, split_factor=1, split_input=True)
    else:
        # input channels are split in half, output channels duplicated
        for name, module in sinks.items():
            _split_channels(
                module,
                channels_to_split,
                grid_aware=grid_aware,
                split_input=True,
                bit_width=weight_bit_width)
        for name, module in sources.items():
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
        split_input: bool,
        weight_bit_width: int) -> GraphModule:
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
        sources = {n: name_to_module[n] for n in region.srcs}
        sinks = {n: name_to_module[n] for n in region.sinks}

        if _is_supported(sources.values(), sinks.values()):
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
                split_input=split_input,
                weight_bit_width=weight_bit_width)

    return model


class RegionwiseChannelSplitting(GraphTransform):

    def __init__(
            self,
            split_ratio=0.02,
            split_criterion='maxabs',
            grid_aware=False,
            split_input=True,
            weight_bit_width=8):
        super(RegionwiseChannelSplitting, self).__init__()

        self.grid_aware = grid_aware
        self.split_ratio = split_ratio
        self.split_criterion = split_criterion
        self.split_input = split_input
        self.weight_bit_width = weight_bit_width

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
                split_input=self.split_input,
                weight_bit_width=self.weight_bit_width)
        if return_regions:
            return self.graph_model, regions
        else:
            return self.graph_model


def split_channels_iteratively(module, split_ratio, grid_aware, bit_width):
    """
    Splits the channels `channels_to_split` of the `weights`.
    `split_input` specifies whether to split Input or Output channels.
    Can also be used to duplicate a channel, just set split_factor to 1.
    Returns: None
    """
    weight = torch.clone(module.weight.data)
    num_channels = weight.shape[1]
    num_channels_to_split = int(math.ceil(split_ratio * num_channels))
    channels_to_split = []
    original_channel_mapping = dict()
    # init scale
    scale = 1.
    for i in range(num_channels_to_split):
        # get the channel to split
        if isinstance(module, nn.Conv2d):
            max_per_channel = weight.abs().max(0).values.flatten(1).max(1).values
        elif isinstance(module, nn.Linear):
            max_per_channel = weight.abs().max(dim=0).values.flatten()
        channels = torch.argsort(max_per_channel, descending=True)
        id = channels[0].item()
        if id < num_channels:
            channels_to_split.append(id)
            # also store the newly created channel to map to the original one
            original_channel_mapping[num_channels + i] = id
        else:
            # one of the newly appended channels has been split, we need to know which original channel that was
            channels_to_split.append(original_channel_mapping[id])
            # also map the new created channel to the original id
            original_channel_mapping[num_channels + i] = original_channel_mapping[id]

        # do channel_splitting
        if isinstance(module, torch.nn.Conv2d):
            # there are four dimensions: [OC, IC, k, k]
            channel = weight[:, id:id + 1, :, :]
            slice1, slice2 = _split_single_channel(channel=channel, grid_aware=False, split_factor=0.5)
            weight = torch.cat((weight[:, :id, :, :], slice1, weight[:, id + 1:, :, :], slice2),
                               dim=1)

        elif isinstance(module, torch.nn.Linear):
            # there are two dimensions: [OC, IC]
            channel = weight[:, id:id + 1]
            slice1, slice2 = _split_single_channel(channel=channel, grid_aware=False, split_factor=0.5)
            weight = torch.cat((weight[:, :id], slice1, weight[:, id + 1:], slice2), dim=1)

    if not grid_aware:
        # set the split weights
        module.weight.data = weight
        if isinstance(module, nn.Conv2d):
            module.in_channels += len(channels_to_split)
        else:
            module.in_features += len(channels_to_split)
        return channels_to_split

    # if grid_aware, calculate the scale of the split weights
    scale = _calculate_scale(weight, bit_width)

    # reset the weight variable
    weight = torch.clone(module.weight.data)

    # reset other variables
    # channels_to_split = []
    original_channel_mapping = dict()

    for i, id in enumerate(channels_to_split):

        if id < num_channels:
            # also store the newly created channel to map to the original one
            original_channel_mapping[num_channels + i] = id
        else:
            # one of the newly appended channels has been split, we need to know which original channel that was
            # also map the new created channel to the original id
            original_channel_mapping[num_channels + i] = original_channel_mapping[id]

        if isinstance(module, torch.nn.Conv2d):
            # there are four dimensions: [OC, IC, k, k]
            channel = weight[:, id:id + 1, :, :]
            slice1, slice2 = _split_single_channel(channel=channel, grid_aware=True, split_factor=0.5, scale=scale)
            weight = torch.cat((weight[:, :id, :, :], slice1, weight[:, id + 1:, :, :], slice2),
                               dim=1)
            module.in_channels += 1

        elif isinstance(module, torch.nn.Linear):
            # there are two dimensions: [OC, IC]
            channel = weight[:, id:id + 1]
            slice1, slice2 = _split_single_channel(channel=channel, grid_aware=True, split_factor=0.5, scale=scale)
            weight = torch.cat((weight[:, :id], slice1, weight[:, id + 1:], slice2), dim=1)
            module.in_features += 1

    module.weight.data = weight

    return channels_to_split


class LayerwiseChannelSplitting(GraphTransform):

    def __init__(
            self,
            split_ratio=0.02,
            split_criterion='maxabs',
            grid_aware=False,
            split_iteratively=False,
            weight_bit_width=8):
        super(LayerwiseChannelSplitting, self).__init__()

        self.grid_aware = grid_aware
        self.split_ratio = split_ratio
        self.split_criterion = split_criterion
        self.weight_bit_width = weight_bit_width
        self.split_iteratively = split_iteratively

    def _is_supported_module(self, graph_model: GraphModule, node: Node) -> bool:
        if node.op == 'call_module':
            module = get_module(graph_model, node.target)
            # so far, only Conv2d and linear layers are supported
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                return True
        return False

    def apply(self, graph_model: GraphModule):
        split_modules = {}
        for node in graph_model.graph.nodes:
            if self._is_supported_module(graph_model, node):
                module = get_module(graph_model, node.target)
                if self.split_iteratively:
                    channels_to_split = split_channels_iteratively(
                        module,
                        split_ratio=self.split_ratio,
                        grid_aware=self.grid_aware,
                        bit_width=self.weight_bit_width)
                else:
                    # we only split input channels
                    channels_to_split = _channels_to_split({}, {node.target: module},
                                                           split_ratio=self.split_ratio,
                                                           split_input=True,
                                                           split_criterion=self.split_criterion)
                    # split the channels in the module
                    _split_channels(
                        module,
                        channels_to_split,
                        grid_aware=self.grid_aware,
                        split_input=True,
                        bit_width=self.weight_bit_width)
                # add node to split modules
                split_modules[module] = torch.tensor(channels_to_split)

        for module, channels_to_split in split_modules.items():
            rewriter = ModuleInstanceToModuleInstance(
                module, ChannelSplitModule(module, channels_to_split))
            rewriter.apply(graph_model)

        return graph_model
