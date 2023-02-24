import torch
import torch.nn as nn

from .mlp import MultiLayerPerceptron

class VariationalEncoder(nn.Module):
    def __init__(
            self,
            input_dim: int = 1000,
            hidden_dim: int = 128,
            z_dim: int = 8,
            activation: nn.Module = nn.ReLU()
    ) -> None:
        super().__init__()

        self.encoder = MultiLayerPerceptron(input_dim, hidden_dim, hidden_dim, 0)
        self.activation = activation
        
        self.logvar_encoder = nn.Linear(hidden_dim, z_dim)
        self.mu_encoder = nn.Linear(hidden_dim, z_dim)

    def forward(self, inputs):
        x = self.encoder(inputs)
        logvar = self.logvar_encoder(x)
        mu = self.mu_encoder(x)

        sigma = torch.exp(0.5 * self.logvar_encoder(x))
        epsilon = torch.randn_like(mu)
        z = mu + epsilon * sigma

        return z, logvar, mu

class VariationalDecoder(nn.Module):
    def __init__(
            self,
            z_dim: int = 8,
            hidden_dim: int = 128,
            x_dim: int = 1000,
            activation: nn.Module = nn.ReLU()
    ) -> None:
        super().__init__()

        self.decoder = MultiLayerPerceptron(z_dim, hidden_dim, x_dim, 0)

    def forward(self, inputs):
        output = self.decoder(inputs)
        return output

if __name__ == "__main__":
    enc = VariationalEncoder()
    dec = VariationalDecoder()

    x = torch.rand((10,1000))

    z, logvar, mu = enc(x)
    assert z.size() == (10, 8), "Shape incorrect"
    assert logvar.size() == (10, 8), "Shape incorrect"
    assert mu.size() == (10, 8), "Shape incorrect"

    recon_x = dec(z)
    assert recon_x.size() == (10, 1000), "Shape incorrect"
