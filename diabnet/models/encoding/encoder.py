from torch import nn

from diabnet.models.encoding.encoders import factory as encoder_factory
from diabnet.models.encoding.pools import factory as pooler_factory


class Encoder(nn.Module):
    """
    Module consists of an encoder and a pooler
    Input dimensions to forward will be B x L x H
    Output dimensions from forward must be B x H.
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.encoder = encoder_factory.get_trajectory_encoder(args.encoder, args)
        self.pooler = pooler_factory.get_trajectory_pooler(args.pooler, hidden_dim=args.hidden_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.pooler(encoded)
        return output
