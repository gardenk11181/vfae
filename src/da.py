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
def domain_adaptation(cfg: DictConfig) -> Tuple[dict, dict]:

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
    src_train_loader = train_loader['source_train']
    tar_train_loader = train_loader['target_train']
    tar_test_loader = datamodule.test_dataloader()
    
    src_train_outputs = trainer.predict(model=model, dataloaders=src_train_loader, ckpt_path=cfg.ckpt_path)
    tar_train_outputs = trainer.predict(model=model, dataloaders=tar_train_loader, ckpt_path=cfg.ckpt_path)
    tar_test_outputs = trainer.predict(model=model, dataloaders=tar_test_loader, ckpt_path=cfg.ckpt_path)

    log.info("Extract z1 and s from trained model")
    src_train_z1 = torch.vstack([output['z1'] for output in src_train_outputs])
    tar_train_z1 = torch.vstack([output['z1'] for output in tar_train_outputs])
    tar_test_z1 = torch.vstack([output['z1'] for output in tar_test_outputs])

    src_train_s = torch.vstack([output['s'] for output in src_train_outputs])
    tar_train_s = torch.vstack([output['s'] for output in tar_train_outputs])
    tar_test_s = torch.vstack([output['s'] for output in tar_test_outputs])

    train_z1 = torch.cat([src_train_z1, tar_train_z1], dim=0).cpu().numpy()
    test_z1 = tar_test_z1.cpu().numpy()

    train_s = torch.cat([src_train_s, tar_train_s], dim=0).reshape(-1).cpu().numpy()
    test_s = tar_test_s.cpu().numpy()

    log.info("Start to train LR and RF")
    lr = LogisticRegression().fit(train_z1, train_s)
    pred_s = lr.predict(test_z1)

    lr_score = np.mean(test_s == pred_s)
    
    log.info(f"lr_score: {lr_score}")

    rf = RandomForestClassifier().fit(train_z1, train_s)
    pred_s = rf.predict(test_z1)

    rf_score = np.mean(test_s == pred_s)

    log.info(f"rf_score: {rf_score}")

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="da.yaml")
def main(cfg: DictConfig) -> None:
    domain_adaptation(cfg)


if __name__ == "__main__":
    main()
