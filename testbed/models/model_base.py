import abc
import enum
from functools import lru_cache, partial
import inspect
import torch
from typing import Callable, Dict, List, Optional, Union

import torch.utils
import torch.utils.hooks


class HookType(enum.Enum):
    TEXT_MODEL_LAYER = enum.auto()
    VISION_MODEL_LAYER = enum.auto()


class ModelBase(abc.ABC):
    def __init__(self, device="auto"):
        """
        Args:
            precision (string): precision used to load model. It can be one of "bf16", "fp16", "fp32", "amp" and "amp_bf16", \
            which represents loading with torch.bfloat16, torch.float16, torch.float32 or auto mixture precision, respectively.
            device (torch.device or str, defaults to None): device that tensors, models, etc. should be moved to. If it is set to \
            "auto", the model should load with `device_map="auto"`, in this case, any `tensor.to(device)` operation will have no effect.
        """

        self.processor = None
        self.model = None
        self.device = device if device != "auto" else None

        self.prompt_template = None

    def _hook_wrapper(self, hook, **hook_args):
        """
        Wraps the provided hook function with additional arguments, if applicable.

        This function checks whether the hook function accepts keyword arguments (**kwargs) or
        explicitly defined arguments that are present in `hook_args`. If the hook supports **kwargs,
        all arguments in `hook_args` are passed to the hook. Otherwise, only the arguments explicitly
        defined in the hook function's signature and present in `hook_args` are passed. If no matching
        arguments exist, the hook is returned unchanged.

        Args:
            hook (Callable): The hook function to be wrapped.
            **hook_args: Additional arguments that may be passed to the hook.

        Returns:
            Callable: A partially applied function if `hook_args` contains matching arguments;
                      otherwise, the original hook function.

        Example:
            If the hook function supports `**kwargs` or specific arguments from `hook_args`:

            >>> def hook_fn(module, input, output, module_name=None):
            >>>     print(f"Module name: {module_name}")

            This wrapper will pass the `module_name` argument from `hook_args` to `hook_fn`.
        """
        signature = inspect.signature(hook)
        supports_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        filtered_args = (
            {k: v for k, v in hook_args.items() if k in signature.parameters}
            if not supports_kwargs
            else hook_args
        )

        if filtered_args:
            return partial(hook, **filtered_args)
        return hook

    def _register_hook(self, register_fn_name, module_name_or_type, hook, **kwargs):
        @lru_cache
        def module_dict():
            return {k: v for k, v in self.model.named_modules()}

        use_regex = kwargs.pop("use_regex", False)
        if use_regex:
            if not isinstance(module_name_or_type, str):
                raise ValueError("module_name_or_type must be string if use_regex=True")
            import re

            return [
                getattr(module, register_fn_name)(
                    self._hook_wrapper(hook, module_name=name), **kwargs
                )
                for name, module in module_dict().items()
                if re.search(module_name_or_type, name)
            ]

        if isinstance(module_name_or_type, HookType):
            raise ValueError(
                f"{module_name_or_type.name} is unsupported or not implemented by {type(self).__name__}."
            )

        return getattr(module_dict()[module_name_or_type], register_fn_name)(
            self._hook_wrapper(hook, module_name=module_name_or_type), **kwargs
        )

    def register_forward_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
        use_regex: bool = False,
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

                    hook(module, args, output, module_name) -> None or modified output
            use_regex (bool, defaults to False):
                If True, `module_name_or_type` will be treated as a regex pattern to match multiple module.

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
            use_regex=use_regex,
        )

    def register_forward_pre_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        use_regex: bool = False,
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

                    hook(module, args, module_name) -> None or modified output
            use_regex (bool, defaults to False):
                If True, `module_name_or_type` will be treated as a regex pattern to match multiple module.

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """

        return self._register_hook(
            "register_forward_pre_hook",
            module_name_or_type,
            hook,
            prepend=prepend,
            with_kwargs=with_kwargs,
            use_regex=use_regex,
        )

    def register_full_backward_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        prepend: bool = False,
        use_regex: bool = False,
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

                    hook(module, grad_input, grad_output, module_name) -> tuple(Tensor) or None
            use_regex (bool, defaults to False):
                If True, `module_name_or_type` will be treated as a regex pattern to match multiple module.

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """
        return self._register_hook(
            "register_full_backward_hook",
            module_name_or_type,
            hook,
            prepend=prepend,
            use_regex=use_regex,
        )

    def register_full_backward_pre_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        prepend: bool = False,
        use_regex: bool = False,
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

                    hook(module, grad_output, module_name) -> tuple[Tensor] or None
            use_regex (bool, defaults to False):
                If True, `module_name_or_type` will be treated as a regex pattern to match multiple module.

        Returns:
            `torch.utils.hooks.RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """

        return self._register_hook(
            "register_full_backward_pre_hook",
            module_name_or_type,
            hook,
            prepend=prepend,
            use_regex=use_regex,
        )

    @property
    def default_prompt_template(self) -> str:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    def process_input(self, *args, **kwargs):
        """
        This function will convert the input (which should be `List[Dict[str, str]]` or `List[str]` for unbatched and
        `List[List[Dict[str, str]]]` or `List[List[str]]` for batched) into the input of models.

        It can be regarded as a composition of `apply_prompt_template` and `processor.__call__`.
        If input is string, `apply_prompt_template` will not be used.
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        """
        A wrapper for genereate method of actual model.
        """
        return self.model.generate(*args, **kwargs)

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
