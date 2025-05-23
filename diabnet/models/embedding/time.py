import torch
from torch import nn


class TimeEmbedder(nn.Module):
    def __init__(self, args, padding_index) -> None:
        super().__init__()
        self.args = args
        self.padding_index = padding_index
        self.add_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)
        self.scale_fc = nn.Linear(args.time_embed_dim, args.hidden_dim)

    def forward(self, x, time_seq: torch.Tensor, padding_mask=None):
        multiplier = 2 * torch.pi / torch.arange(1, self.args.time_embed_dim + 1)

        multiplier = multiplier.reshape(1, len(multiplier)).float().to(self.args.device)
        embed = torch.cos(
            torch.matmul(time_seq.unsqueeze(dim=-1).float(), multiplier)
        )  # Embed Dimension: B x L x H
        if padding_mask is not None:  # Padding mask dimension: B x L
            embed[padding_mask, :] = self.padding_index

        return self.scale_fc(embed) * x + self.add_fc(embed)
