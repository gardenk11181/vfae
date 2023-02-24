import torch
import torch.nn as nn

from .vae import VariationalDecoder, VariationalEncoder

class VariationalFairAutoEncoder(nn.Module):
    def __init__(
            self,
            x_dim: int = 1000,
            s_dim: int = 1,
            y_dim: int = 1,
            z1_dim: int = 50,
            z2_dim: int = 50,
            z1_enc_dim: int = 500, # hidden units of z1 encoder
            z2_enc_dim: int = 300, # hidden units of z2 encoder
            z1_dec_dim: int = 100, # hidden units of z1 decoder
            x_dec_dim: int = 400, # hidden units of x decoder
            activation: nn.Module = nn.ReLU()
    ) -> None:
        super().__init__()

        self.z1_enc = VariationalEncoder(x_dim + s_dim, z1_enc_dim, z1_dim, activation)
        self.z2_enc = VariationalEncoder(z1_dim + y_dim, z2_enc_dim, z2_dim, activation)
        self.z1_dec = VariationalEncoder(z2_dim + y_dim, z1_dec_dim, z1_dim, activation)

        self.x_dec = VariationalDecoder(z1_dim + s_dim, x_dec_dim , x_dim, activation)
        self.y_dec = VariationalDecoder(z1_dim, x_dec_dim, y_dim, activation)

    def forward(self, inputs): #(B, x_dim), (B, s_dim), (B, y_dim) | (B, x_dim), (B, s_dim)
        x = inputs[0]
        s = inputs[1]

        # z1 | x, s ~ N(f(x,s), e^f(x,s))
        x_s = torch.cat([x, s], dim=1)
        z1, z1_mu, z1_logvar = self.z1_enc(x_s)

        # y_dec | z1 ~ Cat(pi = softmax(f(z1)))
        y_recon = self.y_dec(z1)

        # if unsupervised case -> inpute data
        if len(inputs) == 3:
            y = y_recon
        else:
            y = inputs[2]

        # z2 | z1, s ~ N(f(z1,s), e^f(z1,s))
        z1_y = torch.cat([z1, y], dim=1)
        z2, z2_mu, z2_logvar = self.z2_enc(z1_y)

        # z1_dec | z2, y ~ N(f(z2,y), e^f(z2,y))
        z2_y = torch.cat([z2, y], dim=1)
        z1_recon, z1_recon_mu, z1_recon_logvar = self.z1_dec(z2_y)

        # x_dec | z1_dec, s ~ f(z1_dec, s)
        z1_recon_s = torch.cat([z1_recon, s], dim=1)
        x_recon = self.x_dec(z1_recon_s)

        outputs = {
                'z1': z1,
                'z1_mu': z1_mu,
                'z1_logvar': z1_logvar,
                
                'y_recon': y_recon,

                'z2': z2,
                'z2_mu': z2_mu,
                'z2_logaar': z2_logvar,

                'z1_recon': z1_recon,
                'z1_recon_mu': z1_recon_mu,
                'z1_recon_logvar': z1_recon_logvar,

                'x_recon': x_recon,
                }
        return outputs
