import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

import exp_settings as setting
from testbed.models.model_base import HookType


class ICVModel(pl.LightningModule):
    def __init__(self, lmm, icv_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.lmm = lmm
        self.lmm.requires_grad_(False)
        self.icv_encoder = icv_encoder

    def forward(self, ice_texts, query_texts, answers, images):
        pad_token, pad_token_id, eos_token = (
            self.lmm.processor.tokenizer.pad_token,
            self.lmm.processor.tokenizer.pad_token_id,
            self.lmm.processor.tokenizer.eos_token,
        )

        def generate_label_mask(inputs):
            input_ids = inputs["input_ids"]
            batch_size, seq_len = input_ids.shape
            indices = (
                torch.arange(seq_len, device=self.device)
                .unsqueeze(0)
                .expand(batch_size, -1)
            )
            non_pad_mask = input_ids != pad_token_id
            first_non_pad_idx = non_pad_mask.int().argmax(dim=1)

            mask_after_first_non_pad = indices >= first_non_pad_idx.unsqueeze(1)

            separator_pad_mask = mask_after_first_non_pad & ~non_pad_mask
            separator_positions = torch.where(
                separator_pad_mask, indices, torch.full_like(indices, seq_len)
            )
            first_separator_idx = separator_positions.min(dim=1)[0]

            non_answer_mask = (indices >= first_non_pad_idx.unsqueeze(1)) & (
                indices < first_separator_idx.unsqueeze(1)
            )
            return non_pad_mask & ~non_answer_mask
        
        hooks = self.icv_encoder.register_hook_for(self.lmm)

        # step 1. ICV + query + [PAD] + answer [EOS] forward process
        query_answer = [
            query + pad_token + answer + eos_token
            for query, answer in zip(query_texts, answers)
        ]
        query_images = [img[-setting.num_image_in_query :] for img in images]
        query_inputs = self.lmm.process_input(query_answer, query_images).to(
            self.device
        )
        query_inputs["attention_mask"] = query_inputs["input_ids"] != pad_token_id
        query_outputs = self.lmm.model(
            **query_inputs,
            labels=query_inputs["input_ids"],
        )
        icv_logits = query_outputs["logits"]
        for hook in hooks:
            hook.remove()

        # step 2. ICE + query and answer [EOS] forward process
        full_text = [
            ice + query + pad_token + answer + eos_token
            for ice, query, answer in zip(ice_texts, query_texts, answers)
        ]
        inputs = self.lmm.process_input(full_text, images).to(self.device)
        inputs["attention_mask"] = inputs["input_ids"] != pad_token_id
        with torch.no_grad():
            ice_logits = self.lmm.model(**inputs)["logits"]

        # step 3. extract answer logits & calculate kl divergency
        ice_probs = ice_logits[generate_label_mask(inputs)].softmax(dim=-1)
        icv_log_probs = icv_logits[generate_label_mask(query_inputs)].log_softmax(
            dim=-1
        )

        kl_loss = F.kl_div(
            icv_log_probs, ice_probs, reduction="batchmean", log_target=False
        )
        ce_loss = query_outputs["loss"]
        total_loss = kl_loss + setting.ce_loss_weight * ce_loss

        return {
            "kl_loss": kl_loss,
            "ce_loss": ce_loss,
            "loss": total_loss,
        }

    def training_step(self, batch, batch_idx):
        loss_dict = self(**batch)

        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)

        for i, alpha in enumerate(self.icv_encoder.alpha):
            self.log(f"alpha/alpha-{i}", alpha.item())

        return loss_dict["loss"]

    def configure_optimizers(self):
        param_dict = {
            pn: p for pn, p in self.icv_encoder.named_parameters() if p.requires_grad
        }

        alpha_params = [p for n, p in param_dict.items() if "alpha" in n]
        non_alpha_params = [p for n, p in param_dict.items() if not "alpha" in n]

        optim_groups = [
            {"params": non_alpha_params, "lr": setting.icv_lr},
            {"params": alpha_params, "lr": setting.alpha_lr},
        ]

        if "deepspeed" in setting.strategy:
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                weight_decay=setting.weight_decay,
            )
        else:
            optimizer = optim.Adam(
                optim_groups,
                weight_decay=setting.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        warmup_steps = setting.warmup_step
        if isinstance(warmup_steps, float):
            warm_steps = warmup_steps * step_batches
        elif isinstance(warmup_steps, int):
            warm_steps = warmup_steps
        else:
            raise ValueError(
                f"the warm_steps should be int or float, but got {type(warmup_steps)}"
            )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=warm_steps, num_training_steps=step_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if not k.startswith("model")
        }
        return checkpoint
