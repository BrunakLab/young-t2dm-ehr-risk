import torch.nn as nn

from diabnet.models.classifier import factory as classifier_factory


@classifier_factory.RegisterClassifier("Linear")
class LinearRiskClassifier(nn.Module):
    def __init__(self, args, input_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, len(args.month_endpoints))

    def forward(self, x):
        logit = self.linear(x)
        return logit
