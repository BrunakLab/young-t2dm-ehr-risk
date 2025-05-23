import torch
import torch.nn as nn

from diabnet.models.classifier import factory as classifier_factory


@classifier_factory.RegisterClassifier("CumulativeProbability")
class CumulativeProbabilityRiskClassifier(nn.Module):
    """
    The cumulative layer which defines the monotonically increasing risk scores.
    """

    def __init__(self, args, input_dim):
        super().__init__()
        self.args = args
        self.hazard_fc = nn.Linear(input_dim, len(args.month_endpoints))
        self.base_hazard_fc = nn.Linear(input_dim, 1)
        self.relu = nn.ReLU(inplace=True)
        self.max_followup = len(args.month_endpoints)
        mask = torch.ones([len(args.month_endpoints), len(args.month_endpoints)])
        mask = torch.tril(mask, diagonal=0)
        mask = torch.nn.Parameter(torch.t(mask), requires_grad=False)
        self.register_parameter("upper_triagular_mask", mask)

    def hazards(self, x):
        raw_hazard = self.hazard_fc(x)
        pos_hazard = self.relu(raw_hazard)
        return pos_hazard

    def forward(self, x):
        hazards = self.hazards(x)
        B, T = hazards.size()  # hazards is (B, T)
        expanded_hazards = hazards.unsqueeze(-1).expand(B, T, T)  # expanded_hazards is (B,T, T)
        masked_hazards = (
            expanded_hazards * self.upper_triagular_mask
        )  # masked_hazards now (B,T, T)
        cum_prob = torch.sum(masked_hazards, dim=1) + self.base_hazard_fc(x)
        return cum_prob
