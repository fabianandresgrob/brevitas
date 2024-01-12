import torch
from torchvision import models

from brevitas.fx import symbolic_trace
from brevitas.graph.fixed_point import MergeBatchNorm
from brevitas.ptq_algorithms.channel_splitting import *

from .equalization_fixtures import *


@pytest.mark.parametrize('split_ratio', [0.05, 0.1, 0.2])
@pytest.mark.parametrize('split_input', [False, True])
def test_resnet18(split_ratio, split_input):
    model = models.resnet18(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)

    # merge BN before applying channel splitting
    model = MergeBatchNorm().apply(model)

    model = RegionwiseChannelSplitting(
        split_ratio=split_ratio, split_input=split_input).apply(model)
    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)


@pytest.mark.parametrize('split_ratio', [0.05, 0.1])
@pytest.mark.parametrize('split_input', [False, True])
def test_alexnet(split_ratio, split_input):
    model = models.alexnet(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)

    # set split_ratio to 0.2 to def have some splits
    model = RegionwiseChannelSplitting(
        split_ratio=split_ratio, split_input=split_input).apply(model)
    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)


@pytest.mark.parametrize('split_ratio', [0.05, 0.1])
@pytest.mark.parametrize('model_name', ['resnet18', 'alexnet'])
def test_layerwise_splitting(split_ratio, model_name):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=True)

    torch.manual_seed(SEED)
    inp = torch.randn(IN_SIZE_CONV)

    model.eval()
    expected_out = model(inp)
    model = symbolic_trace(model)

    # set split_ratio to 0.2 to def have some splits
    model = LayerwiseChannelSplitting(split_ratio=split_ratio).apply(model)
    out = model(inp)
    assert torch.allclose(expected_out, out, atol=ATOL)
