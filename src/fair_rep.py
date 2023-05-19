from typing import List, Tuple

import hydra
import pyrootutils
from omegaconf import DictConfig

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import numpy as np
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def fair_representation(cfg: DictConfig) -> Tuple[dict, dict]:

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule for train <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    log.info("Starting testing!")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    datamodule.setup('fit')
    train_loader = datamodule.train_dataloader()
    sv_loader = train_loader['supervised']
    usv_loader = train_loader['unsupervised']
    test_loader = datamodule.test_dataloader()
    
    sv_outputs = trainer.predict(model=model, dataloaders=sv_loader, ckpt_path=cfg.ckpt_path)
    usv_outputs = trainer.predict(model=model, dataloaders=usv_loader, ckpt_path=cfg.ckpt_path)
    test_outputs = trainer.predict(model=model, dataloaders=test_loader, ckpt_path=cfg.ckpt_path)

    log.info("Extract z1 and s from trained model")
    sv_z1 = torch.vstack([output['z1'] for output in sv_outputs])
    usv_z1 = torch.vstack([output['z1'] for output in usv_outputs])
    test_z1 = torch.vstack([output['z1'] for output in test_outputs])

    sv_s = torch.vstack([output['s'] for output in sv_outputs])
    usv_s = torch.vstack([output['s'] for output in usv_outputs])
    test_s = torch.vstack([output['s'] for output in test_outputs])

    sv_y = torch.vstack([batch[2] for batch in iter(sv_loader)])
    usv_y = torch.vstack([batch[2] for batch in iter(usv_loader)])
    test_y = torch.vstack([batch[2] for batch in iter(test_loader)])

    train_z1 = torch.cat([sv_z1, usv_z1], dim=0).cpu().numpy()
    test_z1 = test_z1.cpu().numpy()

    train_s = torch.cat([sv_s, usv_s], dim=0).reshape(-1).cpu().numpy()
    test_s = test_s.cpu().numpy()

    train_y = torch.cat([sv_y, usv_y], dim=0).reshape(-1).cpu().numpy()
    test_y = test_y.cpu().numpy()

    log.info("Start to train LR and RF")
    lr = LogisticRegression().fit(train_z1, train_s)
    pred_s = lr.predict(test_z1)

    lr_score = np.mean(test_s == pred_s)
    
    log.info(f"lr_score: {lr_score}")

    rf = RandomForestClassifier().fit(train_z1, train_s)
    pred_s = rf.predict(test_z1)

    rf_score = np.mean(test_s == pred_s)

    log.info(f"rf_score: {rf_score}")

    s0_idx = (test_s == 0).reshape(-1)
    s1_idx = (test_s == 1).reshape(-1)

    lr_y = LogisticRegression().fit(train_z1, train_y)
    pred_y = lr_y.predict(test_z1)
    pred_prob_y = lr_y.predict_proba(test_z1)

    lr_y_score = np.mean(test_y == pred_y)
    
    log.info(f"lr_y_score: {lr_y_score}")

    disc = np.abs(pred_y[s0_idx].mean() - pred_y[s1_idx].mean())
    log.info(f"disc: {disc}")

    disc_prob = np.abs(pred_prob_y[s0_idx].mean() - pred_y[s1_idx].mean())
    log.info(f'disc_prob: {disc_prob}')

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="fair_rep.yaml")
def main(cfg: DictConfig) -> None:
    fair_representation(cfg)


if __name__ == "__main__":
    main()
