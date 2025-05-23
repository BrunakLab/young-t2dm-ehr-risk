import torch
import torch.nn as nn

from diabnet.models.encoding.pools import factory as pooler_factory


@pooler_factory.RegisterTrajectoryPooler("SoftmaxAttention")
class SoftmaxAttentionPooler(nn.Module):
    def __init__(self, hidden_dim, **kwargs):
        super().__init__()
        self.attention_fc = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        x : Tensor where the second to last dimension should be collapsed with respects to attention to the last dimension
        """
        attention_scores = self.attention_fc(x)  # B, N, 1
        attention_scores = self.softmax(attention_scores.squeeze())  # B, 1, N
        x = x * attention_scores.unsqueeze(dim=-1)  # B, C, N
        x = torch.sum(x, dim=-2)
        return x
