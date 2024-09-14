import torch
from torch import nn
import re
import sys
from testbed.models.model_base import HookType


class GlobalICVEncoder(nn.Module):
    def __init__(
        self, lmm_hidden_dim, lmm_layers, alpha_init_value=0.1, **kwargs
    ) -> None:
        super().__init__()

        self.alpha = torch.nn.Parameter(
            torch.full((lmm_layers,), fill_value=alpha_init_value)
        )
        self.icv = torch.nn.Parameter(
            torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
        )

    def register_hook_for(self, lmm, **model_inputs):
        return lmm.register_forward_hook(HookType.TEXT_MODEL_LAYER, self.hook)

    def hook(self, m, inputs, outputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
        print("invoked")
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        hidden_states, *rest = outputs
        shift = self.alpha[layer_idx] * self.icv[layer_idx]
        shifted_states = hidden_states + shift[None, None, :]
        normalized_states = (
            shifted_states
            / shifted_states.norm(dim=-1, keepdim=True)
            * hidden_states.norm(dim=-1, keepdim=True)
        )
        return normalized_states, *rest


class AttnAwareEncoder(nn.Module):
    def __init__(
        self, lmm_hidden_dim, lmm_layers, alpha_init_value=0.1, **kwargs
    ) -> None:
        super().__init__()

        self.alpha = torch.nn.Parameter(
            torch.full((lmm_layers,), fill_value=alpha_init_value)
        )
        self.icv = torch.nn.Parameter(
            torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
        )

    def register_hook_for(self, lmm, **model_inputs):
        return lmm.register_forward_pre_hook(HookType.TEXT_MODEL_LAYER, self.pre_hook)

    def pre_hook(self, m, inputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
        if not isinstance(inputs, tuple):
            inputs = (inputs,)
        hidden_states, *rest = inputs
        shift = self.alpha[layer_idx] * self.icv[layer_idx]
        shifted_states = hidden_states + shift[None, None, :]
        normalized_states = (
            shifted_states
            / shifted_states.norm(dim=-1, keepdim=True)
            * hidden_states.norm(dim=-1, keepdim=True)
        )
        return normalized_states, *rest
