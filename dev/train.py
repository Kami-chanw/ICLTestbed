import os
import shutil

from data_module import DataModule
from shift_encoder import (
    AttnFFNShift,
    ShiftStrategy,
)
from shift_model import ShiftModel, Stratety
from pathlib import Path
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
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)

import hydra
from omegaconf import DictConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@hydra.main(config_path="config", config_name="exp_settings.yaml", version_base=None)
def main(cfg: DictConfig):
    pl.seed_everything(426)
    os.makedirs(config.result_dir, exist_ok=True)
    wb_logger = WandbLogger(
        save_dir=config.result_dir,
        name=cfg.runname,
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
        # fast_dev_run=True,
        max_epochs=10,
        devices=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        use_distributed_sampler=False,
        strategy=cfg.training.strategy,
        precision="16-mixed",
        gradient_clip_val=cfg.training.grad_clip_val,
        log_every_n_steps=10,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        enable_checkpointing=False,
    )
    lmm = Idefics(
        config.idefics_9b_path,
        torch_dtype=torch.float16,
    )

    data_module = DataModule(cfg, lmm)
    shift_encoder = AttnFFNShift(
        4096,
        32,
        attn_strategy=ShiftStrategy.USE_VECTOR_IMPL,
        ffn_strategy=ShiftStrategy.RECORD_HIDDEN_STATES,
    )
    model = ShiftModel(
        cfg,
        lmm,
        shift_encoder,
        Stratety.LAYER_WISE_KL_DIV_AFTER_LM_HEAD | Stratety.LM_LOSS,
    )
    trainer.fit(
        model,
        data_module,
    )
    trainer.save_checkpoint(
        filepath=os.path.join(config.result_dir, "ckpt", "last"),
        weights_only=True,
    )

    if "deepspeed" in cfg.training.strategy:
        convert_zero_ckpt_to_pth(cfg, os.path.join(config.result_dir, "ckpt"))


@rank_zero_only
def convert_zero_ckpt_to_pth(cfg, save_path):
    save_path = Path(save_path)
    cpk_save_path = save_path / "last"
    output_file = save_path / "lightning_module.bin"
    convert_zero_checkpoint_to_fp32_state_dict(cpk_save_path, output_file)

    checkpoint = torch.load(output_file)
    sd = checkpoint["state_dict"]
    sd = {n: pn for n, pn in sd.items() if not n.startswith("lmm")}
    torch.save(sd, save_path / f"{cfg.runname}.pth")
    os.remove(output_file)
    shutil.rmtree(cpk_save_path)


if __name__ == "__main__":
    main()
