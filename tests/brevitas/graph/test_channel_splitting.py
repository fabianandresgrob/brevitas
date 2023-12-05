import torch
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph.fixed_point import MergeBatchNorm
from brevitas.ptq_algorithms.channel_splitting import *

from .equalization_fixtures import *


def test_resnet18():
    model = models.resnet18(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)

    # merge BN before applying channel splitting
    model = MergeBatchNorm().apply(model)

    model = ChannelSplitting(split_ratio=0.1).apply(model)
    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)


def test_alexnet():
    model = models.alexnet(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)

    # set split_ratio to 0.2 to def have some splits
    model = ChannelSplitting(split_ratio=0.2).apply(model)
    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)
