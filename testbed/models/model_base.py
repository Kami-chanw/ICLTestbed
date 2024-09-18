import enum
from functools import lru_cache, partial
import inspect
import re
from typing import Any, Callable, Dict, List, Optional, Union, overload


import torch
from torch.utils.hooks import RemovableHandle
import torch.nn as nn


class HookType(enum.Enum):
    TEXT_MODEL_LAYER = enum.auto()
    VISION_MODEL_LAYER = enum.auto()


class ModelBase(nn.Module):
    def __init__(
        self,
        model_root,
        processor_class,
        model_class,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        super().__init__()

        processor_args = processor_args if processor_args else dict()
        model_args = model_args if model_args else dict()

        self.processor = processor_class.from_pretrained(
            model_root,
            **processor_args,
            **common_args,
        )

        self.model = model_class.from_pretrained(
            model_root, **model_args, **common_args
        )

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

            >>> def hook_fn(module, input, output, module_name):
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

    @overload
    def register_forward_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
        use_regex: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
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
                Note that if use_regex is True, then pattern is not required to match a whole word,
                otherwise module_name_or_type must be exactly the same as the module name.

        Returns:
            `RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_foward_hook(
        self,
        hook,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> RemovableHandle:
        """
        Register a forward hook on this model forward, same as standard one.
        """
        ...

    def register_forward_hook(
        self,
        *args,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
        use_regex: bool = False,
    ):
        if callable(args[0]):
            return super().register_forward_hook(
                *args, prepend=prepend, with_kwargs=with_kwargs, always_call=always_call
            )
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_forward_hook",
                *args,
                prepend=prepend,
                with_kwargs=with_kwargs,
                always_call=always_call,
                use_regex=use_regex,
            )

    @overload
    def register_forward_pre_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        use_regex: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
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
                Note that if use_regex is True, then pattern is not required to match a whole word,
                otherwise module_name_or_type must be exactly the same as the module name.

        Returns:
            `RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_forward_pre_hook(
        self,
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> RemovableHandle:
        """
        Register a forward pre-hook on this model forward, same as standard one.
        """
        ...

    def register_forward_pre_hook(
        self,
        *args,
        prepend: bool = False,
        with_kwargs: bool = False,
        use_regex: bool = False,
    ):
        if callable(args[0]):
            return super().register_forward_pre_hook(
                *args, prepend=prepend, with_kwargs=with_kwargs
            )
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_forward_pre_hook",
                *args,
                prepend=prepend,
                with_kwargs=with_kwargs,
                use_regex=use_regex,
            )

    @overload
    def register_full_backward_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        use_regex: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
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
                Note that if use_regex is True, then pattern is not required to match a whole word,
                otherwise module_name_or_type must be exactly the same as the module name.

        Returns:
            `RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_full_backward_hook(
        self,
        hook: Callable,
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        """
        Register a full_backward hook on this model, same as standard one.
        """
        ...

    def register_full_backward_hook(
        self,
        *args,
        prepend: bool = False,
        use_regex: bool = False,
    ):
        if callable(args[0]):
            return super().register_full_backward_hook(*args, prepend=prepend)
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_full_backward_hook",
                *args,
                prepend=prepend,
                use_regex=use_regex,
            )

    @overload
    def register_full_backward_pre_hook(
        self,
        module_name_or_type: Union[str, HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        use_regex: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
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
                Note that if use_regex is True, then pattern is not required to match a whole word,
                otherwise module_name_or_type must be exactly the same as the module name.

        Returns:
            `RemovableHandle`: a handle that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_full_backward_pre_hook(
        self,
        hook: Callable,
        *,
        prepend: bool = False,
    ) -> RemovableHandle:
        """
        Register a full_backward pre-hook on this model, same as standard one.
        """
        ...

    def register_full_backward_pre_hook(
        self,
        *args,
        prepend: bool = False,
        use_regex: bool = False,
    ):
        if callable(args[0]):
            return super().register_full_backward_pre_hook(*args, prepend=prepend)
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_full_backward_pre_hook",
                *args,
                prepend=prepend,
                use_regex=use_regex,
            )

    @property
    def default_prompt_template(self) -> str:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        raise NotImplementedError

    @property
    def device(self):
        return self.model.device

    def process_input(self, *args, **kwargs):
        """
        This function will convert the input (which should be `List[Dict[str, str]]` or `List[str]` for unbatched and
        `List[List[Dict[str, str]]]` or `List[List[str]]` for batched) into the input of models.

        It can be regarded as a composition of `apply_prompt_template` and `processor.__call__`.
        If input is string, `apply_prompt_template` will not be used.
        """
        raise NotImplementedError

    @torch.no_grad()
    def generate(
        self,
        *inputs,
        processor_args: Dict[str, Any] = None,
        return_inputs: bool = False,
        return_generated_ids: bool = False,
        **generate_args,
    ):
        """
        Generates text using the model based on the provided inputs.

        Args:
            *inputs:
                Inputs that are further fed into `process_input`, see `process_input` docs for details.
            processor_args (Dict[str, Any], optional):
                Additional arguments for the `process_input` method. Defaults to None.
            return_inputs (bool, optional):
                Whether to include the processed inputs in the output dictionary. Defaults to False.
            return_generated_ids (bool, optional):
                Whether to include the generated IDs in the output dictionary. Defaults to False.
            **generate_args:
                Additional arguments to pass to the `generate` method of the model.

        Returns:

            Dict[str, Any]: A dictionary containing:
                - 'outputs': The decoded generated text sequences.
                - 'inputs' (optional): The processed inputs if `return_inputs` is True.
                - 'generated_ids' (optional): The generated token IDs if `return_generated_ids` is True.
        """
        processor_args = processor_args if processor_args else dict()

        inputs = self.process_input(*inputs, **processor_args).to(self.device)
        seq_len = inputs.input_ids.shape[-1]

        generated_ids = self.model.generate(**inputs, **generate_args)
        generated_ids = generated_ids[:, seq_len:]

        outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        if return_inputs == False and return_generated_ids == False:
            return outputs

        result = {"outputs": outputs}
        if return_inputs:
            result["inputs"] = inputs
        if return_generated_ids:
            result["generated_ids"] = generated_ids

        return result

    def forward(
        self, *processor_input, processor_args: Dict[str, Any] = None, **kwargs
    ):
        processor_args = processor_args if processor_args else dict()
        inputs = self.process_input(*processor_input, **processor_args).to(self.device)
        return self.model(**inputs, **kwargs)

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

    from typing import Union, List, Any


import re
import torch.nn as nn


class ModelReplacementMixin:

    @overload
    def replace_module(
        self,
        module_name_or_type: str,
        new_module_cls: type,
        *,
        use_regex: bool = True,
        **init_args,
    ):
        """
        Replaces modules in the model based on module names or types using regex or exact matching,
        with new classes.

        Args:
            module_name_or_type (str or type):
                The module name (str) or type (type) to match.
            new_module_cls (type):
                The new module class to replace the matched module(s).
            use_regex (bool, defaults to True):
                If True, module names are matched using regex.
                If False, performs exact matching.
            **init_args:
                Additional arguments to initialize the new_module_cls.

        Raises:
            ValueError:
                If no matching modules are found in the model.
        """
        pass

    @overload
    def replace_module(
        self,
        module_name_or_type: Union[str, type],
        new_module_instances: Union[Any, List[Any]],
        *,
        use_regex: bool = True,
    ):
        """
        Replaces modules in the model with specific instances. If module_name_or_type is a string and
        use_regex is False, only one module will be matched and new_module_instances must be a single instance.

        Args:
            module_name_or_type (str or type):
                The module name (str) or type (type) to match.
            new_module_instances (Union[Any, List[Any]]):
                A new module instance or list of instances to replace the matched modules.
                If use_regex is False and module_name_or_type is a str, new_module_instances must be a single instance.
            use_regex (bool, defaults to True):
                If True, module names are matched using regex.
                If False, performs exact matching.

        Raises:
            ValueError:
                If the number of matched modules does not equal the number of provided instances.
                If module_name_or_type is str, use_regex is False, and more than one module matches.
        """
        pass

    def replace_module(
        self,
        module_name_or_type: Union[str, type],
        new_module_cls_or_instances: Union[type, Any, List[Any]],
        *,
        use_regex: bool = True,
        **init_args,
    ):
        """
        Replaces modules in the model based on module names or types using regex or exact matching,
        with either new classes or specific instances.

        Args:
            module_name_or_type (str or type):
                The module name (str) or type (type) to match.
            new_module_cls_or_instances (type or Any or List[Any]):
                A new module class to instantiate or a list of new instances to replace the matched modules.
            use_regex (bool, defaults to True):
                If True, module names are matched using regex.
                If False, performs exact matching.
            **init_args:
                Additional arguments to initialize the new_module_cls if provided.

        Raises:
            ValueError:
                If the number of matched modules does not match the number of new instances provided.
                If module_name_or_type is str, use_regex is False, and more than one module matches.
        """
        matched_modules = []

        # Search and replace by name or type
        for name, module in self.model.named_modules():
            if isinstance(module_name_or_type, str):
                # Handle name matching
                if use_regex and re.search(module_name_or_type, name):
                    matched_modules.append((name, module))
                elif not use_regex and name == module_name_or_type:
                    matched_modules.append((name, module))
                    break  # Only one match is allowed for exact match with use_regex=False
            elif isinstance(module, module_name_or_type):
                # Handle type matching
                matched_modules.append((name, module))

        if not matched_modules:
            raise ValueError(f"No modules found matching {module_name_or_type}")

        def replace_module_by_name(name, new_module):
            *parent_module_names, last_name = name.split(".")
            parent_module = self.model
            for pname in parent_module_names:
                parent_module = getattr(parent_module, pname)
            setattr(parent_module, last_name, new_module)

        # Replace with class or instances
        if isinstance(new_module_cls_or_instances, type):
            # Instantiate new modules if given class
            for name, module in matched_modules:
                new_module = new_module_cls_or_instances(**init_args)
                replace_module_by_name(name, new_module)
        else:
            # Replace with provided instances
            if isinstance(new_module_cls_or_instances, list):
                if len(matched_modules) != len(new_module_cls_or_instances):
                    raise ValueError(
                        f"Number of matched modules ({len(matched_modules)}) does not match the number of provided instances ({len(new_module_cls_or_instances)})."
                    )
                for (name, module), new_instance in zip(
                    matched_modules, new_module_cls_or_instances
                ):
                    replace_module_by_name(name, new_instance)
            else:
                # Only one match is allowed when replacing with a single instance
                if len(matched_modules) != 1:
                    raise ValueError(
                        "When replacing with a single instance, only one module should be matched."
                    )
                replace_module_by_name(
                    matched_modules[0][0], new_module_cls_or_instances
                )
