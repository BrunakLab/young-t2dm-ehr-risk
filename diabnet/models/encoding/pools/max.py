import torch
from torch import nn

from diabnet.models.encoding.pools import factory as pooler_factory


@pooler_factory.RegisterTrajectoryPooler("GlobalMax")
class GlobalMaxPooler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        x, _ = torch.max(x, dim=1)
        return x
