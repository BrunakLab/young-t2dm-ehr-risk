import torch.nn as nn

from diabnet.models import factory as model_factory
from diabnet.models.classifier import factory as classfier_factory
from diabnet.models.embedding.code import CodeEmbedder
from diabnet.models.embedding.time import TimeEmbedder
from diabnet.models.encoding.encoder import Encoder


@model_factory.RegisterRiskModel(model_factory.ModelName.SingleRegistryTrajectory)
class SingleRegistryTrajectoryRiskModel(nn.Module):
    """
    Model for single trajectories for discrete risk prediction. Model consists of three parts.
    First part takes the events (tokens and time stamps) from the dataloader and encodes it into a trajectory. Dimension B x L -> B x L x H
    Second part takes the encoded trajectory and creates a hidden representation of this trajectory. Dimension B x L x H -> B x H
    Last part is a risk prediction head. Takes the hidden representation and predicts the output label. Dimension B x H -> B x O
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(p=args.dropout)
        self.embedder = Embedder(args=args)
        self.encoder = Encoder(args=args)

        self.classfier = classfier_factory.get_classifier(
            args.classifier, args=args, input_dim=self.concat_output_dim
        )

    def forward(
        self,
        tokens,
        time_seq=None,
        age_seq=None,
        padding_mask=None,
        mother_outcome=None,
        father_outcome=None,
    ):
        """
        Inputs:
            x: Tensor of code tokens (B x L)
            padding_mask: Tensor mask to mask time and age sequence (B x L)
            time_seq: Tensor of timestamps relative to last event in trajectory (B x L)
            age_seq: Tensor of timestamps relative to birthdate (B x L)
        Outputs:
            logit: Tensor of logits (B x O)
        """
        embeddings = self.embedder(
            tokens=tokens, time_seq=time_seq, age_seq=age_seq, padding_mask=padding_mask
        )
        embeddings = self.dropout(embeddings)
        trajectory_encoded = self.encoder(x=embeddings)
        trajectory_encoded = self.dropout(trajectory_encoded)
        logit = self.classfier(x=trajectory_encoded)
        return logit

    @property
    def inputs(self):
        return [
            arg
            for arg in self.forward.__code__.co_varnames[: self.forward.__code__.co_argcount]
            if arg != "self"
        ]


class Embedder(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args

        if args.use_time_embed and args.encoder != "Transformer":
            print(
                "[W] Time embedding here is designed for transformer only. "
                "But it can work with {} too.".format(args.encoder)
            )

        self.code_embedder = CodeEmbedder(args=args, **kwargs)

        if args.use_time_embed:
            self.time_embedder = TimeEmbedder(args=args, padding_index=0)

        if args.use_age_embed:
            self.age_embedder = TimeEmbedder(args=args, padding_index=0)

        self.batch_norm = nn.BatchNorm1d(args.hidden_dim)

    def forward(self, tokens, time_seq=None, age_seq=None, padding_mask=None):
        x = self.code_embedder(tokens)

        if self.args.use_time_embed:
            time = self.time_embedder(time_seq)
            time[padding_mask] *= 0
            x += time

        if self.args.use_age_embed:
            age = self.age_embedder(age_seq)
            age[padding_mask] *= 0
            x += age

        return x
