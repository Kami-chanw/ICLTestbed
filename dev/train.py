import os

from data_module import ICVDataModule
from global_icv_encoder import GlobalICVEncoder
from icv_model import ICVModel
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
)
from pytorch_lightning.loggers.wandb import WandbLogger
import sys

sys.path.insert(0, "..")
import config
from testbed.models import Idefics, Idefics2
import exp_settings as setting


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


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
            RichProgressBar(),
        ],
        max_epochs=10,
        use_distributed_sampler=False,
        strategy=setting.strategy,
        devices=2,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=25,
        accumulate_grad_batches=1,
        enable_checkpointing=False,
    )
    lmm = Idefics(config.idefics_9b_path)
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
