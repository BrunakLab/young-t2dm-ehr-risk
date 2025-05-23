import torch
from torch import nn


class CodeEmbedder(nn.Module):
    def __init__(self, args, vocab_size, padding_idx) -> None:
        super().__init__()
        self.args = args
        self.padding_idx = padding_idx
        self.code_embed = nn.Embedding(vocab_size, args.hidden_dim, padding_idx=padding_idx)

    def forward(self, tokens: torch.Tensor):
        return self.code_embed(tokens)


class BinnedEmbedder(CodeEmbedder):
    def forward(self, tokens: torch.Tensor):
        embeddings: torch.Tensor
        assert len(tokens.shape) == 3

        normalizer = (tokens != self.padding_idx).sum(dim=2)
        normalizer[normalizer == 0] = 1
        embeddings = self.code_embed(tokens)
        embeddings = embeddings.sum(dim=2) / normalizer.unsqueeze(dim=-1)
        return embeddings


class OneHotCodeEmbedder(nn.Module):
    """
    One-hot embedding for categorical inputs.
    """

    def __init__(self, args, vocab_size, padding_idx):
        super().__init__()
        self.args = args
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(self.vocab_size, self.vocab_size, padding_idx=padding_idx)
        self.embed.weight.data = torch.eye(self.vocab_size)
        self.embed.weight.requires_grad_(False)

    def forward(self, tokens):
        return self.embed(tokens)
