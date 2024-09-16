import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

import exp_settings as setting


class ICVModel(pl.LightningModule):
    def __init__(self, lmm, icv_encoder: torch.nn.Module) -> None:
        super().__init__()
        self.lmm = lmm
        self.lmm.requires_grad_(False)
        self.icv_encoder = icv_encoder

    def generate_label_mask(self, inputs, num_separator, keep_bos=False):
        """
        Generates label mask which masks tokens before num_separator pad_tokens from given inputs.
        """
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        pad_mask = input_ids == self.lmm.processor.tokenizer.pad_token_id
        non_pad_mask = ~pad_mask
        label_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        if self.lmm.processor.tokenizer.padding_side == "left":
            eos_position = non_pad_mask.int().argmax(dim=1)

        for i in range(batch_size):
            seq_pad_positions = pad_mask[i].nonzero(as_tuple=False).squeeze(-1)
            num_pads = len(seq_pad_positions)
            if num_pads < num_separator:
                raise ValueError(
                    f"Sequence {i} has fewer pad tokens ({num_pads}) than num_separator ({num_separator})"
                )

            if self.lmm.processor.tokenizer.padding_side == "left":
                seq_pad_positions = seq_pad_positions[
                    seq_pad_positions > eos_position[i]
                ]

            sep_position = seq_pad_positions[num_separator - 1].item()
            label_mask[i, sep_position + 1 :] = True

        return label_mask & non_pad_mask

    def kl_with_last_logits(self, ice_texts, query_texts, answers, images):
        pad_token, pad_token_id, eos_token = (
            self.lmm.processor.tokenizer.pad_token,
            self.lmm.processor.tokenizer.pad_token_id,
            self.lmm.processor.tokenizer.eos_token,
        )

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

        # step 2. ICE + query + [PAD] answer [EOS] forward process
        full_text = [
            ice + query + pad_token + answer + eos_token
            for ice, query, answer in zip(ice_texts, query_texts, answers)
        ]
        inputs = self.lmm.process_input(full_text, images).to(self.device)
        inputs["attention_mask"] = inputs["input_ids"] != pad_token_id
        with torch.no_grad():
            ice_logits = self.lmm.model(**inputs)["logits"]

        # step 3. extract answer logits & calculate kl divergency
        kl_loss = F.kl_div(
            icv_logits[self.generate_label_mask(query_inputs, 1)].log_softmax(dim=-1),
            ice_logits[self.generate_label_mask(inputs, 1)].softmax(dim=-1),
            reduction="batchmean",
            log_target=False,
        )
        ce_loss = query_outputs["loss"]
        total_loss = kl_loss + setting.ce_loss_weight * ce_loss

        return {
            "kl_loss": kl_loss,
            "ce_loss": ce_loss,
            "loss": total_loss,
        }

    def kl_each_layer(self, ice_texts, query_texts, answers, images):
        pad_token, pad_token_id, bos_token_id, eos_token = (
            self.lmm.processor.tokenizer.pad_token,
            self.lmm.processor.tokenizer.pad_token_id,
            self.lmm.processor.tokenizer.bos_token_id,
            self.lmm.processor.tokenizer.eos_token,
        )

        hooks, record_hooks = self.icv_encoder.register_hook_for(self.lmm)

        # step 1. [SOS](implicitly added) + query + [PAD] + answer [EOS] forward process
        query_answer = [
            query + pad_token + answer + eos_token
            for query, answer in zip(query_texts, answers)
        ]
        query_images = [img[-setting.num_image_in_query :] for img in images]
        query_inputs = self.lmm.process_input(query_answer, query_images).to(
            self.device
        )
        query_inputs["attention_mask"] = query_inputs["input_ids"] != pad_token_id
        self.lmm.model(**query_inputs)
        # peek a hidden_state to deduce shape
        batch_size, _, d_model = self.icv_encoder.hidden_states[0].shape
        query_label_mask = query_inputs["attention_mask"] & (
            query_inputs["input_ids"] != bos_token_id
        )
        icv_hidden_states = (
            torch.cat(self.icv_encoder.hidden_states)
            .masked_select(query_label_mask.unsqueeze(-1))
            .view(batch_size, -1, d_model)
        )
        for hook in hooks:
            hook.remove()

        # step 2. ICE [PAD] query [PAD] answer [EOS] forward process
        full_text = [
            ice + pad_token + query + pad_token + answer + eos_token
            for ice, query, answer in zip(ice_texts, query_texts, answers)
        ]
        inputs = self.lmm.process_input(full_text, images).to(self.device)
        inputs["attention_mask"] = inputs["input_ids"] != pad_token_id
        self.lmm.model(**inputs)["logits"]
        # extract query [PAD] answer [EOS] from hidden_states
        query_label_mask = self.generate_label_mask(inputs, 1)
        ice_hidden_states = (
            torch.cat(self.icv_encoder.hidden_states)
            .masked_select(query_label_mask.unsqueeze(-1))
            .view(batch_size, -1, d_model)
        )
        for hook in record_hooks:
            hook.remove()

        # step 3. calculate kl divergency of each layer
        layer_kl_loss = F.kl_div(
            icv_hidden_states.log_softmax(dim=-1),
            ice_hidden_states.softmax(dim=-1),
            reduction="batchmean",
            log_target=False,
        )

        return {
            "loss": layer_kl_loss,
        }

    def forward(self, ice_texts, query_texts, answers, images):
        if (
            not hasattr(self.icv_encoder, "hidden_states")
            or self.icv_encoder.hidden_states is None
        ):
            self.kl_with_last_logits(ice_texts, query_texts, answers, images)
        return self.kl_each_layer(ice_texts, query_texts, answers, images)

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
