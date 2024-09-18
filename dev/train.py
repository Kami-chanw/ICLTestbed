import os
import shutil

from data_module import DataModule
from dev.shift_encoder import (
    AttnFFNShift,
    AttnShiftFFNLoRA,
)
from dev.shift_model import ShiftModel, Stratety
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

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    pl.seed_everything(426)
    os.makedirs(config.result_dir, exist_ok=True)
    wb_logger = WandbLogger(
        save_dir=config.result_dir,
        name="attn-ffn-lora",
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
        devices=4,
        use_distributed_sampler=False,
        strategy=setting.strategy,
        precision="16-mixed",
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        accumulate_grad_batches=4,
        enable_checkpointing=False,
    )
    lmm = Idefics(
        config.idefics_9b_path,
        torch_dtype=torch.float16,
    )
    icv_encoder = AttnFFNShift(
        4096,
        32,
        attn_shift_enabled=True,
        ffn_shift_enabled=False,
        record_attn_hidden_states=True,
    )
    data_module = DataModule(lmm)
    model = ShiftModel(
        lmm,
        icv_encoder,
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
    torch.save(sd, save_path / "attn-ffn-lora.pth")
    os.remove(output_file)
    shutil.rmtree(cpk_save_path)


if __name__ == "__main__":
    main()
