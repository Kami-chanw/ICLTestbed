import enum
from functools import partial
from typing import List, Callable, Dict, Union
import torch
from torch import nn
import torch.nn.functional as F
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
        self.lmm_hidden_dim = lmm_hidden_dim
        self.lmm_layers = lmm_layers

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

    def register_shift_hooks(self, lmm, **kwargs):
        self_attn_layers = r"model\.layers\.\d+\.self\_attn$"
        mlp_layers = r"model\.layers\.\d+\.mlp$"

        return self.register_hooks(
            lmm,
            "register_forward_hook",
            [self_attn_layers, mlp_layers],
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

    def register_record_hooks(self, lmm, **kwargs):
        self_attn_layers = r"model\.layers\.\d+\.self\_attn$"
        mlp_layers = r"model\.layers\.\d+\.mlp$"

        return self.register_hooks(
            lmm,
            "register_forward_hook",
            [self_attn_layers, mlp_layers],
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


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.mixtral.modeling_mixtral.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def idefics_attn_forward(
    self,
    hidden_states: torch.Tensor,
    key_value_states=None,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions: bool = False,
    use_cache: bool = False,
    module_name=None,
    shift_encoder=None,
):
    # if key_value_states are provided this layer is used as a cross-attention layer
    is_cross_attention = self.is_cross_attention or key_value_states is not None

    bsz, q_len, _ = hidden_states.size()

    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    if not is_cross_attention:
        key_states = (
            self.k_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(hidden_states)
            .view(bsz, q_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
    else:
        _, kv_len, _ = (
            key_value_states.size()
        )  # Note that, in this case, `kv_len` == `kv_seq_len`
        key_states = (
            self.k_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value_states = (
            self.v_proj(key_value_states)
            .view(bsz, kv_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]
    if not is_cross_attention:
        cos, sin = self.rotary_emb(value_states, seq_len=max(kv_seq_len, q_len))
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )
    # [bsz, nh, t, hd]

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    if self.qk_layer_norms:
        query_states = self.q_layer_norm(query_states)
        key_states = self.k_layer_norm(key_states)

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == "cuda" and attention_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
    is_causal = (
        True if self.is_causal and attention_mask is None and q_len > 1 else False
    )

    # ------------------------- The following part is newly added ---------------------

    # calculate Z2 = \sum{ \exp(x_i * \hat{x}^\top) }
    layer_idx = int(re.findall(r"\d+", module_name)[0])
    shift_encoder.Z2[layer_idx] = (
        torch.exp(
            torch.matmul(query_states, key_states.transpose(-2, -1))
            / (self.head_dim**0.5)
        )  # [bsz, nh, t, hd] * [bsz, nh, hd, t] -> [bsz, nh, t, t]
        .sum(dim=-1)  # [bsz, nh, t, t] -> [bsz, nh, t]
        .mean(dim=1)  # [bsz, nh, t] -> [bsz, t]
    )

    # ---------------------------------------------------------------------------------

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.dropout if self.training else 0.0,
        is_causal=is_causal,
    )

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    attn_weights = None

    return attn_output, attn_weights, past_key_value


class AttnApproximator(BaseHookEncoder):
    def __init__(
        self,
        lmm_hidden_dim,
        lmm_layers,
        attn_strategy: ShiftStrategy = ShiftStrategy.USE_VECTOR_IMPL,
        ffn_strategy: ShiftStrategy = None,
        **kwargs,
    ):
        super().__init__()
        self.lmm_hidden_dim = lmm_hidden_dim
        self.lmm_layers = lmm_layers
        self.attn_forward_replaced = False

        def parse_strategy(prefix, strategy):
            params = []

            def load_state_dict_post_hook(module, incompatible_keys):
                for p in params:
                    p.requires_grad_(ShiftStrategy.FROZEN not in strategy)

            if (
                bin(
                    strategy
                    & (ShiftStrategy.USE_VECTOR_IMPL | ShiftStrategy.USE_LORA_IMPL)
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

        # Z1 = \sum{ \exp(x_i X^\top) }
        self.Z1 = torch.nn.Parameter(torch.fill(lmm_layers, 1))
        self.Z2 = [[] for _ in range(lmm_layers)]
        parse_strategy("attn", attn_strategy)
        if ffn_strategy is not None:
            if ffn_strategy != ShiftStrategy.RECORD_HIDDEN_STATES:
                raise ValueError(
                    f"{self.__class__.__name__} only support ShiftStrategy.RECORD_HIDDEN_STATES."
                )
            parse_strategy("ffn", ffn_strategy)

    def register_shift_hooks(self, lmm, **kwargs):
        if "idefics-9b" in lmm.model_name:
            self_attn_layers = r"model\.layers\.\d+\.self\_attn$"
            if self.attn_forward_replaced == False:
                lmm.replace_module_method(
                    self_attn_layers,
                    "forward",
                    partial(idefics_attn_forward, shift_encoder=self),
                    use_regex=True,
                )
                self.attn_forward_replaced = True

            return self.register_hooks(
                lmm,
                "register_forward_hook",
                [self_attn_layers],
                {
                    "attn_hook": (
                        partial(self.attn_shift_hook)
                        if hasattr(self, "attn_lora_A") or hasattr(self, "attn_shift")
                        else None
                    ),
                },
            )
        else:
            raise NotImplementedError(
                f"shift hooks for {lmm.model_name} haven't been implemented yet"
            )

    def register_record_hooks(self, lmm, **kwargs):
        self_attn_layers = r"model\.layers\.\d+\.self\_attn$"
        mlp_layers = r"model\.layers\.\d+\.mlp$"

        return self.register_hooks(
            lmm,
            "register_forward_hook",
            [self_attn_layers, mlp_layers],
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

    def attn_shift_hook(self, m, inputs, outputs, module_name, **kwargs):
        layer_idx = int(re.findall(r"\d+", module_name)[0])
        lora_A = getattr(self, f"attn_lora_A", None)
        lora_B = getattr(self, f"attn_lora_B", None)
        shift = getattr(self, f"attn_shift", None)

        if isinstance(outputs, tuple):
            hidden_states, *rest = outputs
        else:
            hidden_states = outputs

        if lora_A is not None and lora_B is not None:
            shift = (hidden_states @ lora_A[layer_idx]) @ lora_B[layer_idx]
        else:
            shift = shift[layer_idx]
        # \mu = 1 - Z2 / (Z1 + Z2)
        # Z2 shape: (batch_size, seq_len)
        shifted_states = self.do_shift(
            hidden_states,
            (
                1 - self.Z2[layer_idx] / (self.Z1[layer_idx] + self.Z2[layer_idx])
            ).unsqueeze(-1)
            * shift[None, None, :],
        )

        if isinstance(outputs, tuple):
            return (shifted_states, *rest)
        else:
            return shifted_states
