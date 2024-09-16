import torch
from torch import nn
import re
import sys
from testbed.models.model_base import HookType


class GlobalICVEncoder(nn.Module):
    def __init__(
        self, lmm_hidden_dim, lmm_layers, alpha_init_value=0.1, record_hidden_states=False, **kwargs
    ) -> None:
        super().__init__()

        self.alpha = torch.nn.Parameter(
            torch.full((lmm_layers,), fill_value=alpha_init_value)
        )
        self.icv = torch.nn.Parameter(
            torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
        )
        self.hidden_states = [[] for _ in range(lmm_layers)] if record_hidden_states else None 
        

    def register_hook_for(self, lmm, **model_inputs):
        hooks = lmm.register_forward_hook(HookType.TEXT_MODEL_LAYER, self.hook)
        if self.hidden_states:
            return hooks, lmm.register_forward_hook(HookType.TEXT_MODEL_LAYER, self.record_hook)
        return hooks

    def record_hook(self, m, inputs, outputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        hidden_states, *_ = outputs
        self.hidden_states[layer_idx] = hidden_states

    def hook(self, m, inputs, outputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
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
    

class AttnPerturbEncoder(nn.Module):
    def __init__(
        self, lmm_hidden_dim, lmm_layers, alpha_init_value=0.1, record_hidden_states=False, **kwargs
    ) -> None:
        super().__init__()

        self.alpha = torch.nn.Parameter(
            torch.full((lmm_layers,), fill_value=alpha_init_value)
        )
        self.icv = torch.nn.Parameter(
            torch.empty(lmm_layers, lmm_hidden_dim).normal_(mean=0.0, std=0.01)
        )
        self.hidden_states = [[] for _ in range(lmm_layers)] if record_hidden_states else None 

    def register_hook_for(self, lmm, **model_inputs):
        pattern = r"model\.layers\.\d+\.self\_attn$"
        hooks = lmm.register_forward_hook(pattern, self.hook, use_regex=True)
        if self.hidden_states:
            return hooks, lmm.register_forward_hook(pattern, self.record_hook, use_regex=True)
        return hooks
    
    def record_hook(self, m, inputs, outputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        hidden_states, *_ = outputs
        self.hidden_states[layer_idx] = hidden_states

    def hook(self, m, inputs, outputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
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
