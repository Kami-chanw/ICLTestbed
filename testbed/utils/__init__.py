import inspect
import torch
import copy
import types

from functools import partial
from collections.abc import Mapping, Sequence


def try_inject_params(fn, **kwargs):
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
        p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    filtered_args = (
        {k: v for k, v in kwargs.items() if k in signature.parameters}
        if not supports_kwargs
        else kwargs
    )

    if filtered_args:
        return partial(fn, **filtered_args)
    return fn


import torch
import copy
from collections.abc import Mapping, Sequence


def clone_to_device(obj, device="cpu"):
    """
    Recursively traverse an object, clone it, and move all torch.Tensor attributes to a specified device.

    Args:
        obj: The object to process. Can be a dict, list, tuple, set, or a custom class instance.
        device: The device to move the tensors to. Default is 'cpu'.

    Returns:
        A new object with all tensors moved to the specified device.
    """

    def move_to_device(obj, visited):
        obj_id = id(obj)
        if obj_id in visited:
            return obj
        visited.add(obj_id)

        if isinstance(obj, torch.Tensor):
            return obj.clone().to(device)

        elif isinstance(obj, Mapping):
            return {key: move_to_device(value, visited) for key, value in obj.items()}

        elif isinstance(obj, Sequence):
            return type(obj)(move_to_device(item, visited) for item in obj)

        elif hasattr(obj, "__dict__"):
            cloned_obj = copy.copy(obj)
            for attr_name, attr_value in vars(obj).items():
                setattr(cloned_obj, attr_name, move_to_device(attr_value, visited))
            return cloned_obj

        elif hasattr(obj, "__slots__"):
            cloned_obj = copy.copy(obj)
            for slot in obj.__slots__:
                if hasattr(obj, slot):
                    setattr(
                        cloned_obj, slot, move_to_device(getattr(obj, slot), visited)
                    )
            return cloned_obj

        return obj

    return move_to_device(obj, set())
