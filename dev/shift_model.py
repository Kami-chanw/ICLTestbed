import enum
from functools import reduce
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

from shift_encoder import AttnApproximator, AttnFFNShift


class Stratety(enum.IntFlag):
    LAYER_WISE_KL_DIV = 1
    LAYER_WISE_MSE = 2
    LOGITS_KL_DIV = 4
    LM_LOSS = 8
    LAYER_WISE_WEIGHTED = 16
    LAYER_WISE_KL_DIV_AFTER_LM_HEAD = 32 | LAYER_WISE_KL_DIV
    LAYER_WISE_WEIGHTED_MSE = LAYER_WISE_WEIGHTED | LAYER_WISE_MSE
    LAYER_WISE_WEIGHTED_KL_DIV = LAYER_WISE_WEIGHTED | LAYER_WISE_KL_DIV

    def has_layer_wise(self):
        try:
            self.layer_wise_strategy()
            return True
        except ValueError:
            return False

    def validate(self):
        layer_wise_loss = [
            Stratety.LAYER_WISE_KL_DIV,
            Stratety.LAYER_WISE_MSE,
        ]

        if bin(self & reduce(lambda x, y: x | y, layer_wise_loss)).count("1") > 1:
            raise ValueError(
                f"{[e.name for e in layer_wise_loss]} are mutually exclusive."
            )

        if not self.has_layer_wise() and Stratety.LAYER_WISE_WEIGHTED in self:
            raise ValueError(
                "LAYER_WISE_SIM_WEIGHTED should be used with layer wise loss"
            )

    def layer_wise_strategy(self):
        if Stratety.LAYER_WISE_KL_DIV in self:
            return "kl_loss"
        elif Stratety.LAYER_WISE_MSE in self:
            return "mse_loss"
        else:
            raise ValueError("None of layer wise loss strategy is enabled")


class ShiftModel(pl.LightningModule):
    def __init__(self, cfg, lmm, shift_encoder, strategy: Stratety) -> None:
        super().__init__()
        self.lmm = lmm
        self.cfg = cfg
        self.lmm.requires_grad_(False)
        self.shift_encoder = shift_encoder
        strategy.validate()
        self.strategy = strategy
        if Stratety.LAYER_WISE_WEIGHTED in self.strategy:
            self.alpha_linear = nn.ModuleList(
                nn.Linear(shift_encoder.lmm_hidden_dim, 1)
                for _ in range(shift_encoder.lmm_layers)
            )

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
            bos_position = non_pad_mask.int().argmax(dim=1)

        for i in range(batch_size):
            seq_pad_positions = pad_mask[i].nonzero(as_tuple=False).squeeze(-1)

            if self.lmm.processor.tokenizer.padding_side == "left":
                seq_pad_positions = seq_pad_positions[
                    seq_pad_positions > bos_position[i]
                ]

            num_pads = len(seq_pad_positions)
            if num_pads < num_separator:
                raise ValueError(
                    f"Sequence {i} has fewer pad tokens ({num_pads}) than num_separator ({num_separator})"
                )

            sep_position = seq_pad_positions[num_separator - 1].item()
            label_mask[i, sep_position + 1 :] = True

        label_mask = label_mask & non_pad_mask
        if keep_bos:
            label_mask[torch.arange(batch_size, device=self.device), bos_position] = (
                True
            )

        return label_mask

    def remove_hooks(self, hooks):
        # remove all hooks
        for name, handles in hooks.items():
            if isinstance(handles, list):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()

    def get_hidden_states(self, query_label_mask):
        """
        Apply query_label_mask to extract query parts from hidden states (shape: num_layer * [batch_size, seq_len, d_model]),
        and convert to batch_size * [num_layer, query_part_len, d_model].
        """
        hidden_states_dict = {}

        for name, attr in vars(self.shift_encoder).items():
            if "hidden_states" in name:
                # [num_layer, batch_size, seq_len, d_model] -> [batch_size, num_layer, seq_len, d_model]
                hidden_states = torch.stack(attr).transpose(0, 1)
                batch_size, num_layer, seq_len, d_model = hidden_states.shape
                hidden_states_dict[name] = [
                    hs.masked_select(mask[None, :, None]).view(num_layer, -1, d_model)
                    for hs, mask in zip(hidden_states, query_label_mask)
                ]

        if not hidden_states_dict:
            raise RuntimeError("Cannot find any *_hidden_states in shift encoder.")

        return hidden_states_dict

    def calculate_layer_wise_loss(self, icv_hidden_states, ice_hidden_states):
        if Stratety.LAYER_WISE_KL_DIV in self.strategy:

            def kl_div(input, target):
                if Stratety.LAYER_WISE_KL_DIV_AFTER_LM_HEAD in self.strategy:
                    input = self.lmm.model.lm_head(input)
                    target = self.lmm.model.lm_head(target)
                return F.kl_div(
                    input.log_softmax(dim=-1),
                    target.softmax(dim=-1),
                    reduction=(
                        "none"
                        if Stratety.LAYER_WISE_WEIGHTED in self.strategy
                        else "batchmean"
                    ),
                    log_target=False,
                )

            loss_fn = kl_div
        elif Stratety.LAYER_WISE_MSE in self.strategy:
            loss_fn = lambda input, target: F.mse_loss(
                input,
                target,
                reduction=(
                    "none" if Stratety.LAYER_WISE_WEIGHTED in self.strategy else "mean"
                ),
            )

        def criterion(input, target):
            if Stratety.LAYER_WISE_WEIGHTED in self.strategy:
                return torch.mean(
                    (
                        1
                        - torch.mean(
                            F.cosine_similarity(input, target, dim=-1), 1, keepdim=True
                        ).unsqueeze(-1)
                    )
                    * loss_fn(input, target)
                )
            else:
                return loss_fn(input, target)

        layer_loss = dict()
        for (icv_hs_varname, icv_hs_list), (ice_hs_varname, ice_hs_list) in zip(
            icv_hidden_states.items(), ice_hidden_states.items()
        ):
            # hs_list: batch_size * [num_layer, query_part_len, d_model]
            layer_loss[
                icv_hs_varname.replace(
                    "hidden_states", self.strategy.layer_wise_strategy()
                )
            ] = torch.mean(
                torch.stack(
                    [
                        criterion(icv_hs, ice_hs)
                        for icv_hs, ice_hs in zip(icv_hs_list, ice_hs_list)
                    ]
                )
            )
        return layer_loss

    def calculate_logits_kl_loss(
        self, icv_logits, ice_logits, query_label_inputs, ice_label_mask
    ):
        # extract answer [EOS]
        logits_kl_loss = F.kl_div(
            icv_logits[query_label_inputs].log_softmax(dim=-1),
            ice_logits[ice_label_mask].softmax(dim=-1),
            reduction="batchmean",
            log_target=False,
        )
        return {"logits_kl_loss": logits_kl_loss}

    def forward(self, ice_texts, query_texts, answers, images):

        pad_token, pad_token_id, eos_token = (
            self.lmm.processor.tokenizer.pad_token,
            self.lmm.processor.tokenizer.pad_token_id,
            self.lmm.processor.tokenizer.eos_token,
        )
        # prepare inputs
        query_answer = [
            query + pad_token + answer + eos_token
            for query, answer in zip(query_texts, answers)
        ]
        query_images = [img[-self.cfg.data.num_image_in_query :] for img in images]
        query_inputs = self.lmm.process_input(query_answer, query_images).to(
            self.device
        )
        query_inputs["attention_mask"] = query_inputs["input_ids"] != pad_token_id

        full_text = [
            ice + pad_token + query + pad_token + answer + eos_token
            for ice, query, answer in zip(ice_texts, query_texts, answers)
        ]
        inputs = self.lmm.process_input(full_text, images).to(self.device)
        inputs["attention_mask"] = inputs["input_ids"] != pad_token_id

        bos_token_id = self.lmm.processor.tokenizer.bos_token_id
        hooks = self.shift_encoder.register_record_hooks(self.lmm)
        loss_dict = {"loss": 0.0}

        # step 1. [SOS](implicitly added) ICE [PAD] query [PAD] answer [EOS] forward process
        with torch.no_grad():
            ice_logits = self.lmm.model(**inputs)["logits"]

        # extract query + [PAD] + answer + [EOS]
        ice_hidden_states = (
            self.get_hidden_states(self.generate_label_mask(inputs, 1))
            if self.strategy.has_layer_wise()
            else None
        )
        ice_label_mask = self.generate_label_mask(inputs, 2)

        hooks.update(self.shift_encoder.register_shift_hooks(self.lmm))

        # step 2. [SOS](implicitly added) + query + [PAD] + answer [EOS] forward process
        query_outputs = self.lmm.model(
            **query_inputs,
            labels=(
                query_inputs["input_ids"] if Stratety.LM_LOSS in self.strategy else None
            ),
        )
        icv_logits = query_outputs["logits"]

        if Stratety.LM_LOSS in self.strategy:
            loss_dict["ce_loss"] = query_outputs["loss"]
            loss_dict["loss"] += (
                self.cfg.training.ce_loss_weight * query_outputs["loss"]
            )

        # extract query + answer + [EOS]
        icv_hidden_states = (
            self.get_hidden_states(
                query_inputs["attention_mask"]
                & (query_inputs["input_ids"] != bos_token_id)
            )
            if self.strategy.has_layer_wise()
            else None
        )

        self.remove_hooks(hooks)

        # step 3. calculate kl divergency or MSE of each layer
        if self.strategy.has_layer_wise():
            layer_loss = self.calculate_layer_wise_loss(
                icv_hidden_states, ice_hidden_states
            )
            loss_dict.update(layer_loss)
            loss_dict["loss"] += sum(layer_loss.values())

        # step 4. calculate the last logits kl div
        if Stratety.LOGITS_KL_DIV in self.strategy:
            logits_kl_loss = self.calculate_logits_kl_loss(
                icv_logits,
                ice_logits,
                self.generate_label_mask(query_inputs, 1),
                ice_label_mask,
            )
            loss_dict.update(logits_kl_loss)
            loss_dict["loss"] += sum(logits_kl_loss.values())

        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self.forward(**batch)
        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)

        return loss_dict["loss"]

    def configure_optimizers(self):
        param_dict = {
            n: p for n, p in self.shift_encoder.named_parameters() if p.requires_grad
        }
        non_alpha_params = [p for n, p in param_dict.items() if not "alpha" in n]

        optim_groups = [
            {"params": non_alpha_params, "lr": self.cfg.training.icv_lr},
        ]

        if "deepspeed" in self.cfg.training.strategy:
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                weight_decay=self.cfg.training.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                optim_groups,
                weight_decay=self.cfg.training.weight_decay,
            )

        step_batches = self.trainer.estimated_stepping_batches
        warmup_steps = self.cfg.training.warmup_step
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
