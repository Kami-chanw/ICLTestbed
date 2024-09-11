import torch
from torch import nn
import re


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