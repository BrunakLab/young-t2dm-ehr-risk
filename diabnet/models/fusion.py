from torch import nn


class ResidualMlpFusion(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers, dropout) -> None:
        """
        Passing num_layers = 0 and input_dim == ouput_dim will skip this layer
        """
        super().__init__()
        self.num_layers = num_layers
        self.skip_fusion = input_dim == output_dim & num_layers == 0

        self.module = nn.Sequential(nn.Linear(input_dim, output_dim))
        for _ in range(num_layers):
            self.module.append(nn.Dropout(dropout))
            self.module.append(ResidualMlpBlock(output_dim, output_dim))

    def forward(self, x):
        if self.skip_fusion:
            return x

        return self.module(x)


class ResidualMlpBlock(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.gelu = nn.GELU()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.fc2(self.gelu(out))
        out = +x
        return self.gelu(out)
