import torch
from torch import nn

from diabnet.models import factory as model_factory
from diabnet.models.classifier.factory import get_classifier
from diabnet.models.embedding.code import BinnedEmbedder
from diabnet.models.embedding.time import TimeEmbedder
from diabnet.models.encoding.encoder import Encoder
from diabnet.models.encoding.pools.attention import SoftmaxAttentionPooler
from diabnet.utils.vocab import ModalityType


@model_factory.RegisterRiskModel(model_factory.ModelName.MultiRegistryTrajectory)
class BinnedTrajectoryRiskModel(nn.Module):
    def __init__(self, args, vocab_sizes, padding_idxs) -> None:
        super().__init__()
        self.args = args
        self.padding_idxs = padding_idxs
        self.embedders = nn.ModuleDict(
            {
                modality: BinnedEmbedder(
                    args,
                    vocab_size=vocab_sizes[modality],
                    padding_idx=padding_idxs[modality],
                )
                for modality in args.modalities
            }
        )
        self.embedding_pooler = SoftmaxAttentionPooler(self.args.hidden_dim)
        if self.args.use_time_embed:
            self.time_embedder = TimeEmbedder(args, padding_index=0)
        if self.args.use_age_embed:
            self.age_embedder = TimeEmbedder(args, padding_index=0)

        self.encoder = Encoder(args)
        self.encoder_output_dim = self.args.hidden_dim

        self.batch_normalization = nn.BatchNorm1d(self.encoder_output_dim)
        self.classifier = get_classifier(args.classifier, args, input_dim=self.args.hidden_dim)

    def forward(
        self,
        diag_tokens=None,
        prescription_tokens=None,
        ydelse_tokens=None,
        time_seq=None,
        age_seq=None,
        padding_seq=None,
        mother_outcome=None,
        father_outcome=None,
    ):
        embeddings = []
        for modality, tokens in zip(
            [ModalityType.DIAG, ModalityType.PRESCRIPTION, ModalityType.YDELSE],
            [diag_tokens, prescription_tokens, ydelse_tokens],
        ):
            if modality in self.args.modalities:
                try:
                    if modality in self.args.single_index:
                        tokens[tokens != self.padding_idxs[modality]] = 1
                except TypeError:
                    pass

                embeddings.append(self.embedders[modality](tokens=tokens))

        if len(self.args.modalities) > 1:
            # B x L x H x M -> B x L x H
            stacked_embedding = torch.stack(embeddings, dim=-1)
            embedding = self.embedding_pooler(stacked_embedding.transpose(2, 3))
        else:
            embedding = embeddings[0]

        if self.args.use_time_embed:
            embedding = self.time_embedder(
                x=embedding, time_seq=time_seq, padding_mask=padding_seq
            )
        if self.args.use_age_embed:
            embedding = self.age_embedder(x=embedding, time_seq=age_seq, padding_mask=padding_seq)

        encoding = self.encoder(embedding)

        encoding = self.batch_normalization(encoding)

        output = self.classifier(encoding)
        return output

    @property
    def inputs(self):
        return [
            arg
            for arg in self.forward.__code__.co_varnames[: self.forward.__code__.co_argcount]
            if arg != "self"
        ]
