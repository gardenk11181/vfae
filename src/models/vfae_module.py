from typing import Any, List

import torch
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss
import torch.nn.functional as F

from pytorch_lightning import LightningModule

from .components.vfae import VariationalFairAutoEncoder
from ..loss.kl_div import kl_gaussian, kl_bernoulli

class VFAE(LightningModule):
    def __init__(
        self,
        net: VariationalFairAutoEncoder,
        distribution: str,
        alpha: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        self.distribution = distribution
        self.alpha = alpha

        self.ce = CrossEntropyLoss()
        self.bce = BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def log_p(self, x, x_recon):
        if self.distribution == 'poisson':
            log_p = -torch.exp(x_recon) + torch.multiply(x, x_recon)
        elif self.distribution == 'bernoulli':
            pi = torch.sigmoid(x_recon)
            log_p = x * torch.log(pi) + (1-x) * torch.log(1-pi)
        else:
            mu = torch.sigmoid(x_recon)
            logvar = x_recon
            log_p = -0.5 * (x - mu)**2 / torch.exp(logvar) - 0.5 * logvar

        return log_p 

    def supervised_loss(self, batch):
        x, _, y = batch
        outputs = self(batch)

        pred_loss = self.ce(outputs['y_recon'], y)
        log_p = self.log_p(x, outputs['x_recon'])

        z1_kl= kl_gaussian(outputs['z1_mu'], outputs['z1_logvar'], 
                                  outputs['z1_recon_mu'], outputs['z1_recon_logvar'])
        zeros = torch.zeros_like(outputs['z2_mu'])
        z2_kl= kl_gaussian(outputs['z2_mu'], outputs['z2_logvar'],
                                  zeros, zeros)

        loss = -log_p + z1_kl + z2_kl+ self.alpha * pred_loss
        return loss

    def unsupervised_loss(self, batch):
        x, _ = batch
        outputs = self(batch)

        y_pi = F.softmax(outputs['y_recon'])
        y_kl = kl_bernoulli(y_pi, 0.5)
        
        log_p = self.log_p(x, outputs['x_recon'])

        z1_kl= kl_gaussian(outputs['z1_mu'], outputs['z1_logvar'], 
                                  outputs['z1_recon_mu'], outputs['z1_recon_logvar'])
        zeros = torch.zeros_like(outputs['z2_mu'])
        z2_kl= kl_gaussian(outputs['z2_mu'], outputs['z2_logvar'],
                           zeros, zeros)

        loss = -log_p + z1_kl + z2_kl + y_kl

        return loss

    def training_step(self, batch: Any, batch_idx: int):
        sv_loss = self.supervised_loss(batch['source_train'])
        usv_loss = self.unsupervised_loss(batch['target_train'])

        loss = sv_loss + usv_loss
        return {'train_loss': loss, 'supervised_loss': sv_loss, 'unsupervised_loss': usv_loss}

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    net = VariationalFairAutoEncoder()
    optim = torch.optim.Adam
    scheduler = torch.optim.lr_scheduler.StepLR
    _ = VFAE(net,'poisson', 1, optim, scheduler)
