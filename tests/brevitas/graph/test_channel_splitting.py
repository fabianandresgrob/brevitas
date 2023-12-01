import torch
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph.equalize import _extract_regions
from brevitas.ptq_algorithms.channel_splitting import *

from .equalization_fixtures import *


@pytest.mark.skip(reason="Focus on alexnet first")
def test_resnet18():
    model = models.resnet18(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)

    ChannelSplitting(model).apply()
    out = model(inp)
    assert expected_out == out


def test_alexnet():
    model = models.alexnet(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)

    ChannelSplitting(model, split_ratio=0.2).apply()
    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)
