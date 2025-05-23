from torch import nn

from diabnet.models.encoding.encoders import factory as encoder_factory


class AbstractRNNTrajectoryEncoder(nn.Module):
    """
    The abstract risk model which embeds codes using a recurrent neural network (GRU or LSTM).
    Implements the forward method used for these models
    """

    def __init__(self, args):
        super().__init__()
        # Always use bidir RNNs
        assert args.hidden_dim % 2 == 0
        self.hidden_dim = args.hidden_dim // 2

    def forward(self, embed_x):
        seq_hidden, _ = self.rnn(embed_x)
        return seq_hidden


@encoder_factory.RegisterTrajectoryEncoder("GRU")
class GRUTrajectoryEncoder(AbstractRNNTrajectoryEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.rnn = nn.GRU(
            input_size=args.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout,
        )


@encoder_factory.RegisterTrajectoryEncoder("LSTM")
class LSTMTrajectoryEncoder(AbstractRNNTrajectoryEncoder):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.rnn = nn.LSTM(
            input_size=args.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=args.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=args.dropout,
        )
