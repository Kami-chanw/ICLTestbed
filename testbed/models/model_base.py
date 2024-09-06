import abc
import enum
from functools import cached_property, lru_cache
import inspect
import torch
from contextlib import suppress
from typing import Any, Callable, Dict, List, Optional, Union

import torch.utils
import torch.utils.hooks


class HookType(enum.Enum):
    TEXT_MODEL_LAYER = enum.auto()
    VISION_MODEL_LAYER = enum.auto()


class ModelBase(abc.ABC):
    def __init__(self, precision, device="auto"):
        """
        Args:
            precision (string): precision used to load model. It can be one of "bf16", "fp16", "fp32", "amp" and "amp_bf16", \
            which represents loading with torch.bfloat16, torch.float16, torch.float32 or auto mixture precision, respectively.
            device (torch.device or str, defaults to None): device that tensors, models, etc. should be moved to. If it is set to \
            "auto", the model should load with `device_map="auto"`, in this case, any `tensor.to(device)` operation will have no effect.
        """
        precision_dict = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
            "amp": torch.cuda.amp.autocast,
            "amp_bf16": torch.cuda.amp.autocast(dtype=torch.bfloat16),
        }

        self.autocast = precision_dict.get(precision, suppress)
        self.cast_dtype = precision_dict.get(precision, None)

        self.processor = None
        self.model = None
        self.device = device if device != "auto" else None

        self.prompt_template = None

    def _register_hook(self, register_fn_name, module_name_or_type, hook, **kwargs):
        @lru_cache
        def module_dict():
            return {k: v for k, v in self.model.named_modules()}

        register_fn = getattr(module_dict()[module_name_or_type], register_fn_name)
        signature = inspect.signature(register_fn)

        # Remove 'always_call' from kwargs if it's not supported in this version
        if "always_call" in kwargs and "always_call" not in signature.parameters:
            kwargs.pop("always_call")

        return register_fn(hook, **kwargs)

    def register_forward_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> Union[
        torch.utils.hooks.RemovableHandle, List[torch.utils.hooks.RemovableHandle]
    ]:
        """
        Register a forward hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
        for details. The hook will be called every time after forward() has computed an output.

        Args:
            module_name_or_type (str or HookType):
                If str, then call register_forward_hook for the module named.
                If HookType, then register_forward_hook is called for the module of the specified type. It should be implemented by derived classes.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, args, output) -> None or modified output

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """
        return self._register_hook(
            "register_forward_hook",
            module_name_or_type,
            hook,
            prepend=prepend,
            with_kwargs=with_kwargs,
            always_call=always_call,
        )

    def register_forward_pre_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> Union[
        torch.utils.hooks.RemovableHandle, List[torch.utils.hooks.RemovableHandle]
    ]:
        """
        Register a forward pre-hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
        for details. The hook will be called every time before forward() is invoked.

        Args:
            module_name_or_type (str or HookType):
                If str, then call register_forward_pre_hook for the module named.
                If HookType, then register_forward_pre_hook is called for the module of the specified type. It should be implemented by derived classes.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, args) -> None or modified output

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """

        return self._register_hook(
            "register_forward_pre_hook",
            module_name_or_type,
            hook,
            prepend=prepend,
            with_kwargs=with_kwargs,
        )

    def register_full_backward_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        prepend: bool = False,
    ) -> Union[
        torch.utils.hooks.RemovableHandle, List[torch.utils.hooks.RemovableHandle]
    ]:
        """
        Register a full_backward hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
        for details. The hook will be called every time the gradients with respect to a module are computed,
        i.e. the hook will execute if and only if the gradients with respect to module outputs are computed.

        Args:
            module_name_or_type (str or HookType):
                If str, then call register_full_backward_hook for the module named.
                If HookType, then register_full_backward_hook is called for the module of the specified type. It should be implemented by derived classes.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, grad_input, grad_output) -> tuple(Tensor) or None

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """
        return self._register_hook(
            "register_full_backward_hook",
            module_name_or_type,
            hook,
            prepend=prepend,
        )

    def register_full_backward_pre_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        prepend: bool = False,
    ) -> Union[
        torch.utils.hooks.RemovableHandle, List[torch.utils.hooks.RemovableHandle]
    ]:
        """
        Register a full_backward pre-hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook
        for details. The hook will be called every time the gradients for the module are computed.

        Args:
            module_name_or_type (str or HookType):
                If str, then call register_full_backward_pre_hook for the module named.
                If HookType, then register_full_backward_pre_hook is called for the module of the specified type. It should be implemented by derived classes.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, grad_output) -> tuple[Tensor] or None

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """

        return self._register_hook(
            "register_full_backward_pre_hook",
            module_name_or_type,
            hook,
            prepend=prepend,
        )

    @property
    def default_prompt_template(self) -> str:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def generate(self, **kwargs):
        """
        This function will convert the input (which should be `List[Dict[str, Any]]` for unbatched and
        `List[List[Dict[str, Any]]]` for batched) into the actual prompt (using the apply_prompt_template method for example).

        The positional arguments will contain custom arguments and arguments used by generate in transformers.
        If the implementation generates with _generate, extract the custom arguments with _extract_extra_generate_args before doing so.
        """
        raise NotImplementedError

    def _generate(self, args_to_processor, args_to_generate, **kwargs):
        if not isinstance(args_to_processor, dict) or not isinstance(
            args_to_generate, dict
        ):
            raise ValueError(
                "Implement erorr: arguments pass to `ModelBase._generate` should all be dict."
            )

        args_to_processor["padding"] = True
        args_to_processor["return_tensors"] = "pt"

        inputs = self.processor(**args_to_processor).to(
            self.device if self.device is not None else "cuda"
        )

        seq_len = inputs.input_ids.shape[-1]

        generated_ids = self.model.generate(**inputs, **args_to_generate)
        generated_ids = generated_ids[:, seq_len:]

        if kwargs.get("return_inputs", False):
            return (
                self.processor.batch_decode(generated_ids, skip_special_tokens=True),
                inputs,
            )
        else:
            return self.processor.batch_decode(generated_ids, skip_special_tokens=True)

    def _extract_extra_generate_args(self, kwargs):
        """
        Extract key-word args that used in our model generation from a `dict`.
        """
        # if you need more params to control generation, add them here
        params = ["return_inputs"]
        return {key: kwargs.pop(key) for key in params if key in kwargs}

    def apply_prompt_template(
        self,
        conversation: Union[List[Dict[str, str]], List[List[Dict[str, str]]]],
        prompt_template: Optional[str] = None,
        tokenize=False,
        **kwargs,
    ):
        if prompt_template is None:
            prompt_template = (
                self.default_prompt_template
                if self.prompt_template is None
                else self.prompt_template
            )

        if hasattr(self.processor, "apply_chat_template"):
            return self.processor.apply_chat_template(
                conversation, chat_template=prompt_template, tokenize=tokenize, **kwargs
            )

        # all tokenziers have apply_chat_template method
        return self.processor.tokenizer.apply_chat_template(
            conversation, chat_template=prompt_template, tokenize=tokenize, **kwargs
        )
