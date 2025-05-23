import torch
from torch import nn

from diabnet.models import factory as model_factory
from diabnet.models.classifier.factory import get_classifier
from diabnet.utils.vocab import ModalityType


@model_factory.RegisterRiskModel(model_factory.ModelName.BagOfWords)
class BagOfWords(nn.Module):
    def __init__(self, args, vocab_sizes, padding_idxs) -> None:
        super().__init__()
        self.args = args
        self.embedding_output_dim = 0
        self.padding_idxs = padding_idxs
        self.vocab_sizes = vocab_sizes

        for modality in self.args.modalities:
            self.embedding_output_dim += self.vocab_sizes[modality]

        self.classifier = get_classifier(
            args.classifier, args, input_dim=self.embedding_output_dim
        )

    def forward(
        self,
        diag_tokens=None,
        prescription_tokens=None,
        ydelse_tokens=None,
    ):
        embeddings = []
        if ModalityType.DIAG in self.args.modalities:
            embeddings.append(
                self._embed_tokens(
                    diag_tokens,
                    vocab_size=self.vocab_sizes[ModalityType.DIAG],
                )
            )

        if ModalityType.PRESCRIPTION in self.args.modalities:
            embeddings.append(
                self._embed_tokens(
                    prescription_tokens,
                    vocab_size=self.vocab_sizes[ModalityType.PRESCRIPTION],
                )
            )

        if ModalityType.YDELSE in self.args.modalities:
            embeddings.append(
                self._embed_tokens(ydelse_tokens, vocab_size=self.vocab_sizes[ModalityType.YDELSE])
            )
        embedding = torch.cat(embeddings, dim=1)

        output = self.classifier(embedding)
        return output

    def _embed_tokens(self, tokens, vocab_size):
        one_hot = torch.zeros((tokens.shape[0], vocab_size)).to(self.args.device)
        if len(tokens.shape) == 3:
            tokens = tokens.flatten(start_dim=1)
        tokens, _ = tokens.sort(dim=1)
        tokens = torch.unique(tokens, dim=1)
        ones = torch.ones_like(tokens, dtype=torch.float)
        one_hot = one_hot.scatter(index=tokens, src=ones, dim=1)
        return one_hot

    @property
    def inputs(self):
        return [
            arg
            for arg in self.forward.__code__.co_varnames[: self.forward.__code__.co_argcount]
            if arg != "self"
        ]
