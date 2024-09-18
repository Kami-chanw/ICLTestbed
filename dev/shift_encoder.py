from functools import partial
from typing import List, Callable, Dict, Union
import torch
from torch import nn
import re
from testbed.models.model_base import HookType


class BaseHookEncoder(nn.Module):
    def __init__(self, lmm_layers, alpha_init_value=0.1):
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


class AttnFFNShift(BaseHookEncoder):
    def __init__(
        self,
        lmm_hidden_dim,
        lmm_layers,
        alpha_init_value=0.1,
        attn_shift_enabled=True,
        ffn_shift_enabled=True,
        record_attn_hidden_states=False,
        record_ffn_hidden_states=False,
        **kwargs,
    ):
        super().__init__(lmm_layers, alpha_init_value)
        if attn_shift_enabled:
            self.alpha1 = torch.nn.Parameter(
                torch.full((lmm_layers,), fill_value=alpha_init_value)
            )
            self.attn_shift = torch.nn.Parameter(
                torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
            )

        self.attn_hidden_states = (
            [[] for _ in range(lmm_layers)] if record_attn_hidden_states else None
        )

        if ffn_shift_enabled:
            self.alpha2 = torch.nn.Parameter(
                torch.full((lmm_layers,), fill_value=alpha_init_value)
            )
            self.ffn_shift = torch.nn.Parameter(
                torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
            )

        self.ffn_hidden_states = (
            [[] for _ in range(lmm_layers)] if record_ffn_hidden_states else None
        )

        assert attn_shift_enabled or ffn_shift_enabled

    def freeze_attn_shift(self):
        self.alpha1.requires_grad_(False)
        self.attn_shift.requires_grad_(False)

    def unfreeze_attn_shift(self):
        self.alpha1.requires_grad_(True)
        self.attn_shift.requires_grad_(True)

    def freeze_ffn_shift(self):
        self.alpha2.requires_grad_(False)
        self.ffn_shift.requires_grad_(False)

    def unfreeze_ffn_shift(self):
        self.alpha2.requires_grad_(True)
        self.ffn_shift.requires_grad_(True)

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
                "ffn_hook": self.ffn_hook if hasattr(self, "ffn_shift") else None,
                "attn_record_hook": (
                    partial(self.record_hook, record_varname="attn_hidden_states")
                    if self.attn_hidden_states is not None
                    else None
                ),
                "ffn_record_hook": (
                    partial(self.record_hook, record_varname="ffn_hidden_states")
                    if self.ffn_hidden_states is not None
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
            shift = self.alpha2[layer_idx] * self.ffn_shift[layer_idx]
            normalized_states = self.do_shift(hidden_states, shift)
            return (normalized_states, *rest)
        else:
            shift = self.alpha2[layer_idx] * self.ffn_shift[layer_idx]
            normalized_states = self.do_shift(outputs, shift)
            return normalized_states


class AttnShiftFFNLoRA(BaseHookEncoder):
    def __init__(
        self,
        lmm_hidden_dim,
        lmm_layers,
        alpha_init_value=0.1,
        lora_alpha=32,
        r=8,
        attn_shift_enabled=True,
        ffn_shift_enabled=True,
        record_attn_hidden_states=False,
        record_ffn_hidden_states=False,
        **kwargs,
    ):
        super().__init__(lmm_layers, alpha_init_value)
        if attn_shift_enabled:
            self.alpha1 = torch.nn.Parameter(
                torch.full((lmm_layers,), fill_value=alpha_init_value)
            )
            self.attn_shift = torch.nn.Parameter(
                torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
            )
        self.attn_hidden_states = (
            [[] for _ in range(lmm_layers)] if record_attn_hidden_states else None
        )

        if ffn_shift_enabled:
            self.alpha2 = torch.nn.Parameter(
                torch.full((lmm_layers,), fill_value=alpha_init_value)
            )
            self.lora_A = torch.nn.Parameter(torch.randn(lmm_layers, lmm_hidden_dim, r))
            self.lora_B = torch.nn.Parameter(torch.zeros(lmm_layers, r, lmm_hidden_dim))

        self.ffn_hidden_states = (
            [[] for _ in range(lmm_layers)] if record_ffn_hidden_states else None
        )

    def freeze_attn_shift(self):
        self.alpha1.requires_grad_(False)
        self.attn_shift.requires_grad_(False)

    def unfreeze_attn_shift(self):
        self.alpha1.requires_grad_(True)
        self.attn_shift.requires_grad_(True)

    def freeze_ffn_shift(self):
        self.lora_A.requires_grad_(False)
        self.lora_B.requires_grad_(False)

    def unfreeze_ffn_shift(self):
        self.lora_A.requires_grad_(True)
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
                "ffn_hook": self.ffn_hook if hasattr(self, "lora_A") else None,
                "attn_record_hook": (
                    partial(self.record_hook, record_varname="attn_hidden_states")
                    if self.attn_hidden_states is not None
                    else None
                ),
                "ffn_record_hook": (
                    partial(self.record_hook, record_varname="ffn_hidden_states")
                    if self.ffn_hidden_states is not None
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
            shifted_states = (
                hidden_states
                + (hidden_states @ self.lora_A[layer_idx])
                @ self.lora_B[layer_idx]
                * self.alpha2[layer_idx]
            )
            normalized_states = self.do_shift(hidden_states, shifted_states)
            return (normalized_states, *rest)
        else:
            shifted_states = (
                outputs
                + (outputs @ self.lora_A[layer_idx])
                @ self.lora_B[layer_idx]
                * self.alpha2[layer_idx]
            )
            normalized_states = self.do_shift(outputs, shifted_states)
            return normalized_states
