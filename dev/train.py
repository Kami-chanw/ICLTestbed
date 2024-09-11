import os
import shutil
from pathlib import Path

from data_module import ICVDataModule
from global_icv_encoder import GlobalICVEncoder
from icv_model import ICVModel
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import sys

sys.path.insert(0, "..")
import config
from testbed.models import Idefics
from transformers import IdeficsModel


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    pl.seed_everything(436)
    os.makedirs(config.result_dir, exist_ok=True)

    wb_logger = WandbLogger(
        save_dir=config.result_dir,
        name="trial",
        project="VQAInContextVector",
        log_model=False,
    )
    torch.set_float32_matmul_precision("medium")
    trainer = pl.Trainer(
        logger=wb_logger,
        callbacks=[
            LearningRateMonitor(),
            RichModelSummary(max_depth=2),
            RichProgressBar(),
        ],
        max_epochs=10,
        strategy="deepspeed_stage_2_offload",
        precision="bf16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=25,
        accumulate_grad_batches=8,
        enable_checkpointing=False,
    )
    lmm = Idefics(config.idefics_9b_path, dtype=torch.bfloat16)
    icv_encoder = GlobalICVEncoder(4096, 32)
    data_module = ICVDataModule(lmm)
    model = ICVModel(lmm, icv_encoder)
    trainer.fit(
        model,
        data_module,
    )
    trainer.save_checkpoint(
        filepath=os.path.join(
            config.result_dir,
            "ckpt",
            "last.pth",
        ),
        weights_only=True,
    )


if __name__ == "__main__":
    main()
