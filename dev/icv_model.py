from functools import partial
import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup
import exp_settings as setting
from testbed.models.model_base import HookType


class ICVModel(pl.LightningModule):
    def __init__(self, lmm, icv_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.lmm = lmm
        self.icv_encoder = icv_encoder

    def forward(self, ice_texts, query_texts, answers, images):
        tokenizer = self.lmm.processor.tokenizer
        assert tokenizer.padding_side == "left"
        answer_token_lens = torch.tensor(
            [
                len(tokenizer.encode(answer, add_special_tokens=False))
                for answer in answers
            ]
        ).to(self.lmm.device)

        # step 1. ICV + query answer [EOS] forward process
        hooks = self.lmm.register_forward_hook(
            HookType.TEXT_MODEL_LAYER,
            partial(
                self.icv_encoder.hook,
                ice_texts=ice_texts,
                query_texts=query_texts,
                answers=answers,
                images=images,
            ),
        )
        query_answer = [query + answer for query, answer in zip(query_texts, answers)]
        query_images = [img[-setting.num_image_in_query :] for img in images]
        query_inputs = self.lmm.process_input(query_answer, query_images).to(
            self.lmm.device
        )
        query_outputs = self.lmm.model(
            **query_inputs,
            labels=query_inputs["input_ids"],
        )
        icv_logits = query_outputs["logits"]
        for hook in hooks:
            hook.remove()

        # step 2. ICE + query and answer [EOS] forward process
        full_text = [
            ice + query + answer
            for ice, query, answer in zip(ice_texts, query_texts, answers)
        ]
        inputs = self.lmm.process_input(full_text, images).to(self.lmm.device)
        with torch.no_grad():
            ice_logits = self.lmm.model(**inputs)["logits"]

        # step 3. extract answer logits & calculate kl divergency
        max_query_len = query_inputs["input_ids"].shape[1]
        max_input_len = inputs["input_ids"].shape[1]

        query_range = torch.arange(max_query_len, device=answer_token_lens.device)
        input_range = torch.arange(max_input_len, device=answer_token_lens.device)

        zero_shot_mask = query_range[None, :] >= (
            max_query_len - answer_token_lens[:, None] - 1
        )
        icl_context_mask = input_range[None, :] >= (
            max_input_len - answer_token_lens[:, None] - 1
        )

        ice_probs = ice_logits[icl_context_mask].softmax(dim=-1)
        icv_log_probs = icv_logits[zero_shot_mask].log_softmax(dim=-1)

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
            self.log(f"alpha/alpha-{i}", alpha[i])

        return loss_dict["loss"]

    def configure_optimizers(self):
        param_dict = {pn: p for pn, p in self.icv_encoder.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [
            p for n, p in param_dict.items() if p.dim() >= 2 and "alpha" not in n
        ]
        nodecay_params = [
            p for n, p in param_dict.items() if p.dim() < 2 and "alpha" not in n
        ]

        alpha_params = [p for n, p in param_dict.items() if "alpha" in n]

        optim_groups = [
            {"params": decay_params, "weight_decay": setting.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
            {
                "params": alpha_params,
                "weight_decay": setting.weight_decay,
                "lr": setting.alpha_lr,
            },
        ]

        optimizer = DeepSpeedCPUAdam(
            optim_groups,
            lr=setting.icv_lr,
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