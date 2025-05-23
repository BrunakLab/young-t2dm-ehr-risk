from torch import nn

from diabnet.models.encoding.encoders import factory as encoder_factory


@encoder_factory.RegisterTrajectoryEncoder("MLP")
class MLP(nn.Module):
    """
    A basic risk model that embeds codes using a multi-layer perception.
    """

    def __init__(self, args):
        super(MLP, self).__init__()
        self.args = args
        for layer in range(args.num_layers):
            linear_layer = nn.Linear(args.hidden_dim, args.hidden_dim)
            self.add_module("linear_layer_{}".format(layer), linear_layer)
        self.relu = nn.ReLU()

    def forward(self, embed_x):
        for indx in range(self.args.num_layers):
            name = "linear_layer_{}".format(indx)
            embed_x = self._modules[name](embed_x)
            embed_x = self.relu(embed_x)
        return embed_x
