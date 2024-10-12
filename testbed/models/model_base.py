import enum
from functools import lru_cache, partial
import inspect
import re
from types import MethodType
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

        self.config = self.model.config

        self.prompt_template = None

    def _try_inject_params(self, fn, **kwargs):
        """
        Try to inject positional arguments to fn, if applicable.

        This function checks whether the function accepts keyword arguments (**kwargs) or
        explicitly defined arguments that are present in `kwargs`. If the function supports **kwargs,
        all arguments in `kwargs` are passed to the fn. Otherwise, only the arguments explicitly
        defined in the function's signature and present in `kwargs` are passed. If no matching
        arguments exist, the fn is returned unchanged.

        Args:
            fn (Callable): The function to be injected.
            **kwargs: Additional arguments that may be passed to the fn.

        Returns:
            Callable: A partially applied function if `kwargs` contains matching arguments;
                      otherwise, the original function.

        Example:
            If the function supports `**kwargs` or specific arguments from `kwargs`:

            >>> def hook_fn(module, input, output, module_name):
            >>>     print(f"Module name: {module_name}")

            This method will pass the `module_name` argument from `kwargs` to `fn`.
        """
        signature = inspect.signature(fn)
        supports_kwargs = any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        )
        filtered_args = (
            {k: v for k, v in kwargs.items() if k in signature.parameters}
            if not supports_kwargs
            else kwargs
        )

        if filtered_args:
            return partial(fn, **filtered_args)
        return fn

    def _register_hook(self, register_fn_name, module_name_or_type, hook, **kwargs):
        @lru_cache
        def module_dict():
            return {k: v for k, v in self.model.named_modules()}

        if isinstance(module_name_or_type, str):
            # Use regex to match module names
            return [
                getattr(module, register_fn_name)(
                    self._try_inject_params(hook, module_name=name), **kwargs
                )
                for name, module in module_dict().items()
                if re.search(module_name_or_type, name)
            ]

        elif isinstance(module_name_or_type, list):
            # Exact match for each module name in the list
            return [
                getattr(module_dict()[name], register_fn_name)(
                    self._try_inject_params(hook, module_name=name), **kwargs
                )
                for name in module_name_or_type
                if name in module_dict()
            ]

        elif isinstance(module_name_or_type, HookType):
            raise ValueError(
                f"{module_name_or_type.name} is unsupported or not implemented by {type(self).__name__}."
            )

        return getattr(module_dict()[module_name_or_type], register_fn_name)(
            self._try_inject_params(hook, module_name=module_name_or_type), **kwargs
        )

    @overload
    def register_forward_hook(
        self,
        module_name_or_type: Union[str, List[str], HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
        always_call: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
        """
        Register a forward hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_hook
        for details. The hook will be called every time after forward() has computed an output.

        Args:
            module_name_or_type (str, List[str], or HookType):
                If str, then call register_forward_hook for the module named using regex matching.
                If List[str], then register_forward_hook is called for each named module using exact matching.
                If HookType, then register_forward_hook is called for the module of the specified type. It should be implemented by derived classes.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, args, output, module_name) -> None or modified output

        Returns:
            `RemovableHandle` or `List[RemovableHandle]`: a handle or list of handles that can be used to remove the added hook by calling `handle.remove()`
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
            )

    @overload
    def register_forward_pre_hook(
        self,
        module_name_or_type: Union[str, List[str], HookType],
        hook: Callable,
        *,
        prepend: bool = False,
        with_kwargs: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
        """
        Register a forward pre-hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_forward_pre_hook
        for details. The hook will be called every time before forward() is invoked.

        Args:
            module_name_or_type (str, List[str], or HookType):
                If str, then call register_forward_pre_hook for the module named using regex matching.
                If List[str], then register_forward_pre_hook is called for each named module using exact matching.
                If HookType, then register_forward_pre_hook is called for the module of the specified type. It should be implemented by derived classes.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, args, module_name) -> None or modified output

        Returns:
            `RemovableHandle` or `List[RemovableHandle]`: a handle or list of handles that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_forward_pre_hook(
        self, hook: Callable, *, prepend: bool = False, with_kwargs: bool = False
    ) -> RemovableHandle:
        """
        Register a forward pre-hook on this model forward, same as standard one.
        """
        ...

    def register_forward_pre_hook(
        self, *args, prepend: bool = False, with_kwargs: bool = False
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
            )

    @overload
    def register_full_backward_hook(
        self,
        module_name_or_type: Union[str, List[str], HookType],
        hook: Callable,
        *,
        prepend: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
        """
        Register a full backward hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook
        for details. The hook will be called every time the gradients with respect to a module are computed.

        Args:
            module_name_or_type (str, List[str], or HookType):
                If str, then call register_full_backward_hook for the module named using regex matching.
                If List[str], then register_full_backward_hook is called for each named module using exact matching.
                If HookType, then register_full_backward_hook is called for the module of the specified type. It should be implemented by derived classes.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, grad_input, grad_output, module_name) -> None or modified output

        Returns:
            `RemovableHandle` or `List[RemovableHandle]`: a handle or list of handles that can be used to remove the added hook by calling `handle.remove()`
        """
        ...

    @overload
    def register_full_backward_hook(
        self, hook: Callable, *, prepend: bool = False
    ) -> RemovableHandle:
        """
        Register a full_backward hook on this model, same as standard one.
        """
        ...

    def register_full_backward_hook(
        self,
        *args,
        prepend: bool = False,
    ):
        if callable(args[0]):
            return super().register_full_backward_hook(*args, prepend=prepend)
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_full_backward_hook", *args, prepend=prepend
            )

    @overload
    def register_full_backward_pre_hook(
        self,
        module_name_or_type: Union[str, List[str], HookType],
        hook: Callable,
        *,
        prepend: bool = False,
    ) -> Union[RemovableHandle, List[RemovableHandle]]:
        """
        Register a full backward pre-hook on the module, see https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_pre_hook
        for details. The hook will be called every time before the gradients with respect to a module are computed.

        Args:
            module_name_or_type (str, List[str], or HookType):
                If str, then call register_full_backward_pre_hook for the module named using regex matching.
                If List[str], then register_full_backward_pre_hook is called for each named module using exact matching.
                If HookType, then register_full_backward_pre_hook is called for the module of the specified type. It should be implemented by derived classes.
            hook (Callable):
                The user defined hook to be registered. The hook should have the following signature::

                    hook(module, grad_output, module_name) -> None or modified output

        Returns:
            `RemovableHandle` or `List[RemovableHandle]`: a handle or list of handles that can be used to remove the added hook by calling `handle.remove()`
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

    def register_full_backward_pre_hook(self, *args, prepend: bool = False):
        if callable(args[0]):
            return super().register_full_backward_pre_hook(*args, prepend=prepend)
        elif len(args) >= 2 and callable(args[1]):
            return self._register_hook(
                "register_full_backward_pre_hook", *args, prepend=prepend
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

    @overload
    def replace_module(
        self,
        module_name_or_type: Union[str, List[str], nn.Module],
        new_module_cls: nn.Module,
        *,
        strict: bool = True,
        **init_args,
    ):
        """
        Replace modules in the model by matching their names or types, using either regex
        for string input or exact string matching for a list of strings.

        Args:
            module_name_or_type (str, List[str], or nn.Module):
                Name(s) or type of the module to be replaced. If str, regex matching is used.
                If List[str], exact string matching is performed.
            new_module_cls (nn.Module):
                The new module class to replace the matched module(s).
            strict (bool, defaults to True):
                If True, checks that the new function's signature is compatible with the old method's signature.
                If False, skips signature checking.
            **init_args:
                Arguments to initialize the new module.

        Raises:
            ValueError:
                If no matching modules are found, or if the new module's forward method
                has incompatible parameter names with the original module.
        """
        pass

    @overload
    def replace_module(
        self,
        module_name_or_type: Union[str, List[str], nn.Module],
        new_module_instances: Union[nn.Module, List[nn.Module]],
        *,
        strict: bool = True,
    ):
        """
        Replace specific instances of modules in the model by matching their names or types.
        If `module_name_or_type` is a string, regex matching is used. If it is a list of strings,
        exact string matching is performed.

        Args:
            module_name_or_type (str, List[str], or nn.Module):
                Name(s) or type of the module to be replaced. If str, regex matching is used.
                If List[str], exact string matching is performed.
            new_module_instances (Union[nn.Module, List[nn.Module]]):
                New module instance(s) to replace the matched modules.
            strict (bool, defaults to True):
                If True, checks that the new function's signature is compatible with the old method's signature.
                If False, skips signature checking.

        Raises:
            ValueError:
                If the number of matched modules doesn't match the number of provided instances,
                or if the new module's forward method has incompatible parameter names with the original module.
        """
        pass

    def replace_module(
        self,
        module_name_or_type: Union[str, List[str], nn.Module],
        new_module_cls_or_instances: Union[nn.Module, List[nn.Module]],
        *,
        strict: bool = True,
        **init_args,
    ):
        """
        Replace modules in the model based on names or types, with either new classes or specific instances.
        Supports matching by regex when `module_name_or_type` is a string, or exact string matching when
        it is a list of strings. It checks whether the `forward` method of the new module has compatible
        parameter names with the original module's `forward` method.

        Args:
            module_name_or_type (str, List[str], or nn.Module):
                Name(s) or type of the module to be replaced. If str, regex matching is used.
                If List[str], exact string matching is performed.
            new_module_cls_or_instances (nn.Module or List[nn.Module]):
                A new module class or instance(s) to replace the matched modules.
            strict (bool, defaults to True):
                If True, checks that the new function's signature is compatible with the old method's signature.
                If False, skips signature checking.
            **init_args:
                Additional arguments to initialize the new module if providing a class.

        Raises:
            ValueError:
                If the number of matched modules does not match the number of new instances provided,
                or if the new module's forward method has incompatible parameter names with the original module.
        """

        def replace_module_by_name(name, orig_module, new_module):
            if strict:
                orig_params = list(
                    inspect.signature(orig_module.forward).parameters.keys()
                )
                new_params = list(
                    inspect.signature(new_module.forward).parameters.keys()
                )

                if orig_params != new_params[: len(orig_params)]:
                    raise ValueError(
                        "The first few parameters of the new module's forward method do not match "
                        "the original module's. If you want to add new parameters, they should be "
                        "at the end of the parameter list."
                    )

            *parent_module_names, last_name = name.split(".")
            parent_module = self.model
            for pname in parent_module_names:
                parent_module = getattr(parent_module, pname)
            setattr(parent_module, last_name, new_module)

        if isinstance(module_name_or_type, str):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if re.search(module_name_or_type, name)
            }
        elif isinstance(module_name_or_type, list):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if name in module_name_or_type
            }
        elif isinstance(module_name_or_type, type):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if isinstance(module, module_name_or_type)
            }
        else:
            matched_modules = None

        if not matched_modules:
            raise ValueError(f"No modules found matching {module_name_or_type}")

        if isinstance(new_module_cls_or_instances, type):
            # Create new instances using the provided class and replace the matched modules
            for name, module in matched_modules.items():
                replace_module_by_name(
                    name, module, new_module_cls_or_instances(**init_args)
                )
        else:
            if isinstance(new_module_cls_or_instances, list):
                if len(matched_modules) != len(new_module_cls_or_instances):
                    raise ValueError(
                        f"Number of matched modules ({len(matched_modules)}) does not match the number of provided instances ({len(new_module_cls_or_instances)})."
                    )
                for (name, module), new_instance in zip(
                    matched_modules.items(), new_module_cls_or_instances
                ):
                    replace_module_by_name(name, module, new_instance)
            else:
                if len(matched_modules) != 1:
                    raise ValueError(
                        "When replacing with a single instance, only one module should be matched."
                    )
                replace_module_by_name(
                    next(iter(matched_modules)),
                    matched_modules[next(iter(matched_modules))],
                    new_module_cls_or_instances,
                )

    def replace_module_method(
        self,
        module_name_or_type: Union[str, List[str], nn.Module],
        method_name: str,
        new_method: Callable,
        *,
        strict: bool = True,
    ):
        """
        Replace a method of modules in the model based on names or types, with a new function.
        Optionally checks whether the new function's signature is compatible with the old method's signature.
        The new function will have `module_name` and `old_method` injected as keyword arguments, if possible.

        Args:
            module_name_or_type (str, List[str], or nn.Module):
                Name(s) or type of the module whose method is to be replaced. If str, regex matching is used. 
                If List[str], exact string matching is performed.
            method_name (str):
                The name of the method to replace (e.g., 'forward').
            new_method (Callable):
                A new function to replace the matched method.
            strict (bool, defaults to True):
                If True, checks that the new function's signature is compatible with the old method's signature.
                If False, skips signature checking.

        Raises:
            ValueError:
                If no module matches the given name or type, or if the new method's signature is incompatible 
                with the old method's signature when strict is True.
        """
        if isinstance(module_name_or_type, str):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if re.search(module_name_or_type, name)
            }
        elif isinstance(module_name_or_type, list):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if name in module_name_or_type
            }
        elif isinstance(module_name_or_type, type):
            matched_modules = {
                name: module
                for name, module in self.model.named_modules()
                if isinstance(module, module_name_or_type)
            }
        else:
            matched_modules = None

        if not matched_modules:
            raise ValueError(f"No modules found matching {module_name_or_type}")

        for name, module in matched_modules.items():
            old_method = getattr(module, method_name, None)
            if old_method is None:
                raise ValueError(f"Module '{name}' has no method '{method_name}'")

            if strict:
                orig_params = list(inspect.signature(old_method).parameters.keys())
                new_params = list(inspect.signature(new_method).parameters.keys())

                if "self" not in orig_params and isinstance(old_method, MethodType):
                    orig_params.insert(0, "self")

                if orig_params != new_params[: len(orig_params)]:
                    raise ValueError(
                        f"The first few parameters of the new function do not match "
                        f"the original method '{method_name}' of module '{name}'."
                    )

            setattr(
                module,
                method_name,
                MethodType(
                    self._try_inject_params(
                        new_method, module_name=name, old_method=old_method
                    ),
                    module,
                ),
            )

