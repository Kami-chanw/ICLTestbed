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
import exp_settings as setting
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--runname', type=str, required=True, help='Name of the run')
args = parser.parse_args()

def main():
    pl.seed_everything(426)
    os.makedirs(config.result_dir, exist_ok=True)
    wb_logger = WandbLogger(
        save_dir=config.result_dir,
        name=args.runname,
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
        strategy=setting.strategy,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        accumulate_grad_batches=setting.accumulate_grad_batches,
        enable_checkpointing=False,
    )
    lmm = Idefics(
        config.idefics_9b_path,
        torch_dtype=torch.float16,
    )

    data_module = DataModule(lmm)
    shift_encoder = AttnFFNShift(
        4096,
        32,
        attn_strategy=ShiftStrategy.USE_VECTOR_IMPL,
        ffn_strategy=ShiftStrategy.RECORD_HIDDEN_STATES,
    )
    model = ShiftModel(
        lmm,
        shift_encoder,
        Stratety.LAYER_WISE_MSE,
    )
    trainer.fit(
        model,
        data_module,
    )
    trainer.save_checkpoint(
        filepath=os.path.join(config.result_dir, "ckpt", "last"),
        weights_only=True,
    )

    if "deepspeed" in setting.strategy:
        convert_zero_ckpt_to_pth(os.path.join(config.result_dir, "ckpt"))


@rank_zero_only
def convert_zero_ckpt_to_pth(save_path):
    save_path = Path(save_path)
    cpk_save_path = save_path / "last"
    output_file = save_path / "lightning_module.bin"
    convert_zero_checkpoint_to_fp32_state_dict(cpk_save_path, output_file)

    checkpoint = torch.load(output_file)
    sd = checkpoint["state_dict"]
    sd = {n: pn for n, pn in sd.items() if not n.startswith("lmm")}
    torch.save(sd, save_path / f"{args.runname}.pth")
    os.remove(output_file)
    shutil.rmtree(cpk_save_path)


if __name__ == "__main__":
    main()
