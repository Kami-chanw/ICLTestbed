import enum
from functools import partial
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import get_cosine_schedule_with_warmup

import exp_settings as setting


class Stratety(enum.IntFlag):
    LAYER_WISE_KL_DIV = 1
    LAYER_WISE_MSE = 2
    LOGITS_KL_DIV = 4
    LM_LOSS = 8
    ALTERNATE_TRAINING = 16

    def is_layer_wise(self):
        return Stratety.LAYER_WISE_KL_DIV in self or Stratety.LAYER_WISE_MSE in self

    def layer_wise_strategy(self):
        if Stratety.LAYER_WISE_KL_DIV in self:
            return "kl_loss"
        elif Stratety.LAYER_WISE_MSE in self:
            return "mse_loss"
        else:
            raise ValueError("None of layer wise loss strategy is enabled")


class ShiftModel(pl.LightningModule):
    def __init__(self, lmm, icv_encoder: torch.nn.Module, strategy: Stratety) -> None:
        super().__init__()
        if (
            Stratety.LAYER_WISE_KL_DIV in strategy
            and Stratety.LAYER_WISE_MSE in strategy
        ):
            raise ValueError("Layer wise kl loss and mse loss are mutually exclusive.")
        self.lmm = lmm
        self.lmm.requires_grad_(False)
        self.icv_encoder = icv_encoder
        self.strategy = strategy

    def generate_label_mask(self, inputs, num_separator):
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

        return label_mask & non_pad_mask

    def forward(self, ice_texts, query_texts, answers, images):
        pad_token, pad_token_id, bos_token_id, eos_token = (
            self.lmm.processor.tokenizer.pad_token,
            self.lmm.processor.tokenizer.pad_token_id,
            self.lmm.processor.tokenizer.bos_token_id,
            self.lmm.processor.tokenizer.eos_token,
        )

        def get_hidden_states(query_label_mask):
            """
            Apply query_label_mask to extract query parts from hidden states (shape: num_layer * [batch_size, seq_len, d_model]),
            and convert to batch_size * [num_layer, query_part_len, d_model].
            """
            hidden_states_dict = {}

            for name, attr in vars(self.icv_encoder).items():
                if "hidden_states" in name and attr is not None:
                    # [num_layer, batch_size, seq_len, d_model] -> [batch_size, num_layer, seq_len, d_model]
                    hidden_states = torch.stack(attr).permute(1, 0, 2, 3)
                    batch_size, num_layer, seq_len, d_model = hidden_states.shape
                    hidden_states_dict[name] = [
                        hs.masked_select(mask[None, :, None]).view(
                            num_layer, -1, d_model
                        )
                        for hs, mask in zip(hidden_states, query_label_mask)
                    ]

            if not hidden_states_dict:
                raise RuntimeError(
                    "Cannot find any *_hidden_states in shift encoder. Did you forget set record_*_hidden_states to True in __init__ of shift encoder?"
                )

            return hidden_states_dict

        loss_dict = {"loss": 0.0}
        hooks = self.icv_encoder.register_hook_for(self.lmm)

        # step 1. [SOS](implicitly added) + query + answer [EOS] forward process
        query_answer = [
            query + answer + eos_token for query, answer in zip(query_texts, answers)
        ]
        query_images = [img[-setting.num_image_in_query :] for img in images]
        query_inputs = self.lmm.process_input(query_answer, query_images).to(
            self.device
        )
        # query_inputs["attention_mask"] = query_inputs["input_ids"] != pad_token_id

        if Stratety.LM_LOSS in self.strategy:
            query_inputs["labels"] = query_inputs["input_ids"]

        query_outputs = self.lmm.model(**query_inputs)
        icv_logits = query_outputs["logits"]

        if Stratety.LM_LOSS in self.strategy:
            loss_dict["ce_loss"] = query_outputs["loss"]
            loss_dict["loss"] += query_outputs["loss"]

        if self.strategy.is_layer_wise() or Stratety.LOGITS_KL_DIV in self.strategy:
            # remove bos_token, model cannot predict bos_token
            query_label_mask = query_inputs["attention_mask"].bool() & (
                query_inputs["input_ids"] != bos_token_id
            )

        # hidden states need to be recorded only when layer-wise comparison is enabled
        if self.strategy.is_layer_wise():
            icv_hidden_states = get_hidden_states(query_label_mask)

        # remove all shift hooks
        for name, handles in hooks.items():
            if "record" not in name:
                if isinstance(handles, list):
                    for handle in handles:
                        handle.remove()
                else:
                    handles.remove()
        hooks = {
            name: hook_fn for name, hook_fn in hooks.items() if "record" not in name
        }

        # step 2. [SOS](implicitly added) ICE query [PAD] answer [EOS] forward process
        full_text = [
            ice + pad_token + query + answer + eos_token
            for ice, query, answer in zip(ice_texts, query_texts, answers)
        ]
        inputs = self.lmm.process_input(full_text, images).to(self.device)
        inputs["attention_mask"] = inputs["input_ids"] != pad_token_id
        with torch.no_grad():
            ice_logits = self.lmm.model(**inputs)["logits"]

        if self.strategy.is_layer_wise() or Stratety.LOGITS_KL_DIV in self.strategy:
            ice_label_mask = self.generate_label_mask(inputs, 1)

        if self.strategy.is_layer_wise():
            ice_hidden_states = get_hidden_states(ice_label_mask)

        for name, handles in hooks.items():
            if isinstance(handles, list):
                for handle in handles:
                    handle.remove()
            else:
                handles.remove()

        # step 3. calculate kl divergency or MSE of each layer
        if self.strategy.is_layer_wise():

            if Stratety.LAYER_WISE_KL_DIV in self.strategy:

                def kl_div(input, target):
                    return F.kl_div(
                        input.log_softmax(dim=-1),
                        target.softmax(dim=-1),
                        reduction="batchmean",
                        log_target=False,
                    )

                loss_fn = kl_div
            elif Stratety.LAYER_WISE_MSE in self.strategy:
                loss_fn = F.mse_loss

            layer_loss = dict()
            for (icv_hs_varname, icv_hs_list), (ice_hs_varname, ice_hs_list) in zip(
                icv_hidden_states.items(), ice_hidden_states.items()
            ):
                layer_loss[
                    icv_hs_varname.replace(
                        "hidden_states", self.strategy.layer_wise_strategy()
                    )
                ] = torch.mean(
                    torch.stack(
                        [
                            loss_fn(icv_hs, ice_hs)
                            for icv_hs, ice_hs in zip(icv_hs_list, ice_hs_list)
                        ]
                    )
                )
            loss_dict.update(layer_loss)
            loss_dict["loss"] += sum(layer_loss.values())

        # step 4. calculate the last logits kl div
        if Stratety.LOGITS_KL_DIV in self.strategy:
            logits_kl_loss = F.kl_div(
                icv_logits[query_label_mask].log_softmax(dim=-1),
                ice_logits[ice_label_mask].softmax(dim=-1),
                reduction="batchmean",
                log_target=False,
            )
            loss_dict["logits_kl_loss"] = logits_kl_loss
            loss_dict["loss"] += logits_kl_loss

        return loss_dict

    def training_step(self, batch, batch_idx):
        loss_dict = self(**batch)

        self.log_dict(loss_dict, sync_dist=True, prog_bar=True)

        for name, param in self.icv_encoder.named_parameters():
            if name.startswith("alpha"):
                for i, a in enumerate(param):
                    self.log(f"alpha/{name}-{i}", a.item())

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
