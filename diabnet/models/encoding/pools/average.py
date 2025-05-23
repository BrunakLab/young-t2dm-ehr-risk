import torch
import torch.nn as nn

from diabnet.models.encoding.pools import factory as pooler_factory


@pooler_factory.RegisterTrajectoryPooler("GlobalAverage")
class GlobalAveragePooler(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, x):
        spatially_flat_size = (*x.size()[:2], -1)
        x = x.view(spatially_flat_size)
        x = torch.mean(x, dim=1)
        return x
