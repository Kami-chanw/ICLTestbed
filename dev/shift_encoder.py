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
            shift = shift.unsqueeze(0)
        if shift.dim() < 3:
            shift = shift.unsqueeze(0)

        shifted_states = hidden_states + shift

        return shifted_states

    def record_hook(self, m, inputs, outputs, module_name, record_varname, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        hidden_states, *_ = outputs
        getattr(self, record_varname)[layer_idx] = hidden_states


class ShiftStrategy(enum.IntFlag):
    USE_VECTOR_IMPL = 1
    USE_LORA_IMPL = 2
    USE_STRANGE_ATTN_IMPL = 4
    RECORD_HIDDEN_STATES = 8
    FROZEN = 16


class AttnFFNShift(BaseHookEncoder):
    def __init__(
        self,
        lmm_hidden_dim,
        lmm_layers,
        attn_strategy: ShiftStrategy = None,
        ffn_strategy: ShiftStrategy = None,
        **kwargs,
    ):
        super().__init__()

        def parse_strategy(prefix, strategy):
            params = []

            def load_state_dict_post_hook(module, incompatible_keys):
                for p in params:
                    p.requires_grad_(ShiftStrategy.FROZEN not in strategy)

            if (
                bin(
                    strategy
                    & (
                        ShiftStrategy.USE_VECTOR_IMPL
                        | ShiftStrategy.USE_LORA_IMPL
                        | ShiftStrategy.USE_STRANGE_ATTN_IMPL
                    )
                ).count("1")
                > 1
            ):
                raise ValueError(
                    "The shift implementation methods are mutually exclusive."
                )

            def add_attr(name, value):
                setattr(self, name, value)
                params.append(getattr(self, name))

            if ShiftStrategy.USE_VECTOR_IMPL in strategy:
                add_attr(
                    f"{prefix}_shift",
                    torch.nn.Parameter(
                        torch.empty(lmm_layers, lmm_hidden_dim).normal_(
                            mean=0.0, std=0.001
                        )
                    ),
                )
            elif ShiftStrategy.USE_LORA_IMPL in strategy:
                r = kwargs.get("r", 8)
                add_attr(
                    f"{prefix}_lora_A",
                    torch.nn.Parameter(torch.randn(lmm_layers, lmm_hidden_dim, r)),
                )
                add_attr(
                    f"{prefix}_lora_B",
                    torch.nn.Parameter(torch.zeros(lmm_layers, r, lmm_hidden_dim)),
                )

            if ShiftStrategy.RECORD_HIDDEN_STATES in strategy:
                setattr(
                    self, f"{prefix}_hidden_states", [[] for _ in range(lmm_layers)]
                )

            self.register_load_state_dict_post_hook(load_state_dict_post_hook)

        if attn_strategy is not None:
            parse_strategy("attn", attn_strategy)
        if ffn_strategy is not None:
            parse_strategy("ffn", ffn_strategy)

    def attn_shift_params(self):
        for n, p in self.named_parameters():
            if n.startswith("attn"):
                yield n, p

    def ffn_shift_params(self):
        for n, p in self.named_parameters():
            if n.startswith("ffn"):
                yield n, p

    def freeze_attn_shift(self):
        for name, params in self.attn_shift_params():
            params.requires_grad_(False)

    def unfreeze_attn_shift(self):
        for name, params in self.attn_shift_params():
            params.requires_grad_(True)

    def freeze_ffn_shift(self):
        for name, params in self.ffn_shift_params():
            params.requires_grad_(False)

    def unfreeze_ffn_shift(self):
        for name, params in self.ffn_shift_params():
            params.requires_grad_(True)

    def register_shift_hooks(self, lmm, **model_inputs):
        self_attn_layers = r"model\.layers\.\d+\.self\_attn$"
        mlp_layers = r"model\.layers\.\d+\.mlp$"

        return self.register_hooks(
            lmm,
            "register_forward_hook",
            [
                self_attn_layers,
                mlp_layers,
            ],
            {
                "attn_hook": (
                    self._shift_hook("attn")
                    if hasattr(self, "attn_lora_A") or hasattr(self, "attn_shift")
                    else None
                ),
                "ffn_hook": (
                    self._shift_hook("ffn")
                    if hasattr(self, "ffn_lora_A") or hasattr(self, "ffn_shift")
                    else None
                ),
            },
        )

    def register_record_hooks(self, lmm, **model_inputs):
        self_attn_layers = r"model\.layers\.\d+\.self\_attn$"
        mlp_layers = r"model\.layers\.\d+\.mlp$"

        return self.register_hooks(
            lmm,
            "register_forward_hook",
            [
                self_attn_layers,
                mlp_layers,
            ],
            {
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

    def _shift_hook(self, prefix):
        def hook(m, inputs, outputs, module_name, **kwargs):
            layer_idx = int(re.findall(r"\d+", module_name)[0])
            lora_A = getattr(self, f"{prefix}_lora_A", None)
            lora_B = getattr(self, f"{prefix}_lora_B", None)
            shift = getattr(self, f"{prefix}_shift", None)

            if isinstance(outputs, tuple):
                hidden_states, *rest = outputs
            else:
                hidden_states = outputs

            if lora_A is not None and lora_B is not None:
                shift = (hidden_states @ lora_A[layer_idx]) @ lora_B[layer_idx]
            else:
                shift = shift[layer_idx]
            shifted_states = self.do_shift(hidden_states, shift)

            if isinstance(outputs, tuple):
                return (shifted_states, *rest)
            else:
                return shifted_states

        return hook
