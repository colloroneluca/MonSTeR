import os
import random
import numpy as np
import torch
import pytorch_lightning as pl

from tqdm import tqdm
import logging
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import wandb

from src.config import read_config, save_config, pop_key
from src.data.collate import collate_text_motion
from src.logging import log_config, get_run_name
from src.load import load_pretrained

logger = logging.getLogger(__name__)


def save_rng_state(path="rng_state.pt"):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
    }
    torch.save(state, path)
    print(f"Saved RNG state to {path}")


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig):
    ckpt = None
    if cfg.resume_dir is not None:
        assert cfg.ckpt is not None
        ckpt = cfg.ckpt
        cfg = read_config(cfg.resume_dir)
        logger.info("Resuming training")
        logger.info(f"The config is loaded from: \n{cfg.resume_dir}")
    else:
        config_path = save_config(cfg)
        logger.info("Training script")
        logger.info(f"The config can be found here: \n{config_path}")

    SEED = cfg.seed
    print("Using", SEED)

    pl.seed_everything(SEED, workers=True)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # required for CUDA determinism
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    debug = pop_key(cfg, "debug")
    logger.info(f"Debug: {debug}")
    offline_logging = pop_key(cfg, "offline_logging")
    ckpt_path = pop_key(cfg.model, "ckpt_name")
    if debug:
        os.environ["WANDB_DISABLED"] = "true"
    if offline_logging:
        os.environ["WANDB_MODE"] = "offline"
    cfg.trainer["logger"]["name"] = get_run_name(cfg)
    tags = ["MonSTeR"]
    tags = tags + [p for p in cfg.trainer["logger"]["name"].split("|")]
    cfg.trainer["logger"]["tags"] = tags
    cfg.trainer.logger.project = "MonSTeR"

    cfg.trainer.logger.project = "MonSTeR"
    if "id" in cfg.keys():
        run = wandb.init(
            entity="your_entity", project="MonSTeR", id=cfg.id, resume="must"
        )

    train_dataset = instantiate(cfg.data, split="train", debug=debug)
    val_dataset = instantiate(cfg.data, split="val", debug=debug, preload=False)

    train_dataloader = instantiate(
        cfg.dataloader_train,
        dataset=train_dataset,
        collate_fn=collate_text_motion,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    val_dataloader = instantiate(
        cfg.dataloader_val,
        dataset=val_dataset,
        pin_memory=True,
        collate_fn=collate_text_motion,
        shuffle=False,
    )

    # NOTE: modifying the total number of epochs/steps will affect deterministic behaviour
    logger.info("Loading the model")
    cfg.model["train_dl_len"] = len(train_dataloader)
    cfg.model["total_steps"] = len(train_dataloader) * cfg.trainer["max_epochs"]
    model = instantiate(cfg.model)

    try:
        model = load_pretrained(model, cfg, ckpt_path)
        logger.info(f"Loaded model from {ckpt_path}")
    except:
        logger.info("ATTENTION! Pretrained weights are not being loaded!")

    trainer = instantiate(cfg.trainer)

    if trainer.local_rank == 0:
        save_config(cfg, path=trainer.logger.experiment.dir)

    if trainer.local_rank == 0:
        log_config(config_path=config_path)

    logger.info("Training")
    trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=ckpt)


if __name__ == "__main__":
    train()
