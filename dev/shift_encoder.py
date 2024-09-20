import enum
from functools import partial
from typing import List, Callable, Dict, Union
import torch
from torch import nn
import re
from testbed.models.model_base import HookType


class BaseHookEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def register_hooks(
        self,
        lmm,
        register_fn_name: str,
        targets: List[Union[str, HookType]],
        hooks: Dict[str, Callable],
    ):
        handles = dict()

        for target, (name, hook_fn) in zip(targets, hooks.items()):
            if hook_fn is not None and not "record" in name:
                handles[name] = getattr(lmm, register_fn_name)(
                    target, hook_fn, use_regex=True
                )

        # all record hooks should be called after shift hooks
        for target, (name, hook_fn) in zip(targets, hooks.items()):
            if hook_fn is not None and "record" in name:
                handles[name] = getattr(lmm, register_fn_name)(
                    target, hook_fn, use_regex=True
                )

        return handles

    def do_shift(self, hidden_states, shift: torch.Tensor):
        if shift.dim() < 2:
            shift.unsqueeze_(0)
        if shift.dim() < 3:
            shift.unsqueeze_(0)

        shifted_states = hidden_states + shift
        normalized_states = (
            shifted_states / shifted_states.norm(dim=-1, keepdim=True)
        ) * hidden_states.norm(dim=-1, keepdim=True)
        return normalized_states

    def record_hook(self, m, inputs, outputs, module_name, record_varname, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        hidden_states, *_ = outputs
        getattr(self, record_varname)[layer_idx] = hidden_states


class ShiftConfig(enum.IntFlag):
    ATTENTION_SHIFT = 1
    FFN_SHIFT = 2
    FFN_LORA = 4
    RECORD_ATTN = 8
    RECORD_FFN = 16


class ShiftConfig:
    def __init__(
        self, lmm_hidden_dim, lmm_layers, strategy, alpha_init_value=0.1, **kwargs
    ):
        self.lmm_hidden_dim = lmm_hidden_dim
        self.lmm_layers = lmm_layers
        self.strategy = strategy
        self.alpha_init_value = alpha_init_value

        if ShiftConfig.FFN_LORA in strategy:
            self.r = kwargs.get("r", 8)

        assert strategy & (ShiftConfig.ATTENTION_SHIFT | ShiftConfig.FFN_SHIFT)

        # fmt: off
        if bin(strategy & (ShiftConfig.FFN_LORA | ShiftConfig.FFN_SHIFT)).count('1') > 1:
            raise ValueError("ffn lora and ffn shift are mutually exclusive.")
        # fmt: on


class AttnFFNShift(BaseHookEncoder):
    def __init__(self, shift_config: ShiftConfig):
        super().__init__()

        lmm_layers, lmm_hidden_dim, alpha_init_value, strategy = (
            shift_config.lmm_layers,
            shift_config.lmm_hidden_dim,
            shift_config.alpha_init_value,
            shift_config.strategy,
        )

        if ShiftConfig.ATTENTION_SHIFT in strategy:
            self.alpha1 = torch.nn.Parameter(
                torch.full((lmm_layers,), fill_value=alpha_init_value)
            )
            self.attn_shift = torch.nn.Parameter(
                torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
            )

        if ShiftConfig.RECORD_ATTN in strategy:
            self.attn_hidden_states = [[] for _ in range(lmm_layers)]

        if ShiftConfig.FFN_SHIFT in strategy or ShiftConfig.FFN_LORA in strategy:
            self.alpha2 = torch.nn.Parameter(
                torch.full((lmm_layers,), fill_value=alpha_init_value)
            )
            if ShiftConfig.FFN_LORA in strategy:
                r = shift_config.r
                self.lora_A = torch.nn.Parameter(
                    torch.randn(lmm_layers, lmm_hidden_dim, r)
                )
                self.lora_B = torch.nn.Parameter(
                    torch.zeros(lmm_layers, r, lmm_hidden_dim)
                )
            else:
                self.ffn_shift = torch.nn.Parameter(
                    torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
                )

        if ShiftConfig.RECORD_FFN in strategy:
            self.ffn_hidden_states = [[] for _ in range(lmm_layers)]

    def freeze_attn_shift(self):
        if hasattr(self, "alpha1"):
            self.alpha1.requires_grad_(False)
        if hasattr(self, "attn_shift"):
            self.attn_shift.requires_grad_(False)

    def unfreeze_attn_shift(self):
        if hasattr(self, "alpha1"):
            self.alpha1.requires_grad_(True)
        if hasattr(self, "attn_shift"):
            self.attn_shift.requires_grad_(True)

    def freeze_ffn_shift(self):
        if hasattr(self, "alpha2"):
            self.alpha2.requires_grad_(False)
        if hasattr(self, "ffn_shift"):
            self.ffn_shift.requires_grad_(False)
        if hasattr(self, "lora_A"):
            self.lora_A.requires_grad_(False)
        if hasattr(self, "lora_B"):
            self.lora_B.requires_grad_(False)

    def unfreeze_ffn_shift(self):
        if hasattr(self, "alpha2"):
            self.alpha2.requires_grad_(True)
        if hasattr(self, "ffn_shift"):
            self.ffn_shift.requires_grad_(True)
        if hasattr(self, "lora_A"):
            self.lora_A.requires_grad_(True)
        if hasattr(self, "lora_B"):
            self.lora_B.requires_grad_(True)

    def register_hook_for(self, lmm, **model_inputs):
        self_attn_layers = r"model\.layers\.\d+\.self\_attn$"
        mlp_layers = r"model\.layers\.\d+\.mlp$"

        return self.register_hooks(
            lmm,
            "register_forward_hook",
            [
                self_attn_layers,
                mlp_layers,
                self_attn_layers,
                mlp_layers,
            ],
            {
                "attn_hook": self.attn_hook if hasattr(self, "attn_shift") else None,
                "ffn_hook": (
                    self.ffn_hook
                    if hasattr(self, "lora_A") or hasattr(self, "ffn_shift")
                    else None
                ),
                "attn_record_hook": (
                    partial(self.record_hook, record_varname="attn_hidden_states")
                    if hasattr(self, "attn_hidden_states")
                    else None
                ),
                "ffn_record_hook": (
                    partial(self.record_hook, record_varname="ffn_hidden_states")
                    if hasattr(self, "ffn_hidden_states")
                    else None
                ),
            },
        )

    def attn_hook(self, m, inputs, outputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])

        if isinstance(outputs, tuple):
            hidden_states, *rest = outputs
            shift = self.alpha1[layer_idx] * self.attn_shift[layer_idx]
            normalized_states = self.do_shift(hidden_states, shift)
            return (normalized_states, *rest)
        else:
            shift = self.alpha1[layer_idx] * self.attn_shift[layer_idx]
            normalized_states = self.do_shift(outputs, shift)
            return normalized_states

    def ffn_hook(self, m, inputs, outputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])

        if isinstance(outputs, tuple):
            hidden_states, *rest = outputs
            if hasattr(self, "lora_A"):
                shifted_states = (
                    hidden_states
                    + (hidden_states @ self.lora_A[layer_idx])
                    @ self.lora_B[layer_idx]
                    * self.alpha2[layer_idx]
                )
            else:
                shift = self.alpha2[layer_idx] * self.ffn_shift[layer_idx]
                shifted_states = self.do_shift(hidden_states, shift)

            return (shifted_states, *rest)
        else:
            if hasattr(self, "lora_A"):
                shifted_states = (
                    outputs
                    + (outputs @ self.lora_A[layer_idx])
                    @ self.lora_B[layer_idx]
                    * self.alpha2[layer_idx]
                )
            else:
                shift = self.alpha2[layer_idx] * self.ffn_shift[layer_idx]
                shifted_states = self.do_shift(outputs, shift)

            return shifted_states
