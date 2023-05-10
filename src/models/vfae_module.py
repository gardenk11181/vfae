from typing import Any, List

import torch
torch.set_float32_matmul_precision('high')
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss

from pytorch_lightning import LightningModule

from .components.vfae import VariationalFairAutoEncoder
from ..loss.kl_div import kl_gaussian, kl_bernoulli
from ..loss.hsic import hsic
from ..loss.mmd import fast_mmd

class VFAE(LightningModule):
    def __init__(
        self,
        net: VariationalFairAutoEncoder,
        distribution: str,
        alpha: int,
        lamb_mmd: int,
        lamb_hsic: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net'])

        self.net = net
        self.distribution = distribution
        self.alpha = alpha
        self.lamb_mmd = lamb_mmd
        self.lamb_hsic = lamb_hsic

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

        log_p = torch.sum(log_p, dim=1)
        return log_p 

    def supervised_loss(self, batch):
        x, _, y = batch
        outputs = self(batch)

        pred_loss = self.bce(outputs['y_recon'], y)
        log_p = self.log_p(x, outputs['x_recon'])

        z1_kl= kl_gaussian(outputs['z1_mu'], outputs['z1_logvar'], 
                                  outputs['z1_recon_mu'], outputs['z1_recon_logvar'])
        zeros = torch.zeros_like(outputs['z2_mu'])
        z2_kl= kl_gaussian(outputs['z2_mu'], outputs['z2_logvar'],
                                  zeros, zeros)

        loss = -log_p + z1_kl + z2_kl+ self.alpha * pred_loss
        total_loss = torch.sum(loss, dim=0)

        return total_loss, outputs['z1']

    def unsupervised_loss(self, batch):
        x, _ = batch
        outputs = self(batch)

        y_pi = torch.sigmoid(outputs['y_recon'])
        one_half = torch.tensor([0.5], dtype=torch.float32).to(self.device)
        y_kl = kl_bernoulli(y_pi, one_half.repeat([x.shape[0]]).reshape(-1,1)).reshape(-1)

        log_p = self.log_p(x, outputs['x_recon']).reshape(-1)

        z1_kl= kl_gaussian(outputs['z1_mu'], outputs['z1_logvar'], 
                                  outputs['z1_recon_mu'], outputs['z1_recon_logvar']).reshape(-1)
        zeros = torch.zeros_like(outputs['z2_mu'])
        z2_kl= kl_gaussian(outputs['z2_mu'], outputs['z2_logvar'],
                           zeros, zeros).reshape(-1)

        loss = -log_p + z1_kl + z2_kl + y_kl
        total_loss = torch.sum(loss, dim=0)

        return total_loss, outputs['z1']

    def training_step(self, batch: Any, batch_idx: int):
        sv_loss, sv_z1 = self.supervised_loss(batch['source_train'])
        usv_loss, usv_z1 = self.unsupervised_loss(batch['target_train'])

        z1 = torch.cat([sv_z1, usv_z1], dim=0)
        s = torch.cat([batch['source_train'][1], batch['target_train'][1]], dim=0)
        mmd_loss = fast_mmd(sv_z1, usv_z1)
        hsic_loss = hsic(z1, s)

        loss = sv_loss + usv_loss + self.lamb_mmd * mmd_loss + self.lamb_hsic * hsic_loss

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('supervised_loss', sv_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('unsupervised_loss', usv_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('mmd_loss', mmd_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('hsic_loss', hsic_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {'loss': loss, 'supervised_loss': sv_loss, 'unsupervised_loss': usv_loss}

    def test_step(self, batch: Any, batch_idx: int):
        outputs = self(batch)
            
        y_pi = torch.sigmoid(outputs['y_recon'])
        pred_y = torch.round(y_pi)
        
        correct = pred_y == batch[2]
        return {'correct': correct}

    def test_epoch_end(self, test_step_outputs):
        sum = 0
        count = 0
        for out in test_step_outputs:
            sum += torch.sum(out['correct'])
            count += len(out['correct'])
        acc_y = sum / count

        self.log('acc_y', acc_y, prog_bar=True)
        return acc_y

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train_loss",
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
