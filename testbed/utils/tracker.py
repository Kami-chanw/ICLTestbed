import torch.nn as nn
import weakref

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from torch.utils.hooks import RemovableHandle
from functools import wraps

from . import try_inject_params, clone_to_device

__all__ = ["ForwardTracker", "GradTracker"]


@dataclass
class ModuleStatus:
    module_name: str
    accessed: bool


class TrackerBase(ABC):
    """
    The base class of all trackers.
    """

    def __init__(self):
        super().__init__()
        self._handles: Tuple[RemovableHandle]
        self._module_refs_dict: weakref.WeakKeyDictionary
        self._data: Dict[str, List[List]]
        self._tracker_fn: Callable

    @abstractmethod
    def _register_tracker(self, module: nn.Module) -> RemovableHandle:
        raise NotImplementedError()

    def _hook_wrapper(self, hook):
        """
        Wrap the hook to:
            1. Call user defined tracker fn, and inject module_name if the tracker is attched to a model.
            2. If it is a new forward pass, append a new list for all the keys in `_data`.
        """

        @wraps(hook)
        def wrapper(*args, **kwargs):
            m = args[0]
            tracker_fn = (
                try_inject_params(
                    self._tracker_fn, module_name=self._module_refs_dict[m].module_name
                )
                if self._module_refs_dict[m].module_name
                else self._tracker_fn
            )
            tracker_fn(*args, **kwargs)
            if self._module_refs_dict[m].accessed:
                for k in self._data:
                    self._data[k].append([])
                for status in self._module_refs_dict.values():
                    status.accessed = False

            hook(*args, **kwargs)

            self._module_refs_dict[m].accessed = True

        return wrapper

    def track(self, modules: List[nn.Module]):
        """
        Track a list of modules.

        Args:
            modules (List[nn.Module]):
                A list of modules to track.
        """
        if not isinstance(modules, list) or not isinstance(modules[0], nn.Module):
            raise TypeError(
                f"modules should be a list of nn.Module, but got {type(modules)}"
            )

        if hasattr(self, "_handles"):
            self.remove()

        self._module_refs_dict = weakref.WeakKeyDictionary(
            {
                # module_name will be assigned when the tracker is attached to a model
                m: ModuleStatus(module_name=None, accessed=False)
                for m in modules
            }
        )
        self._handles = tuple(self._register_tracker(m) for m in modules)

    def remove(self) -> None:
        for h in self._handles:
            h.remove()

        del self._handles
        del self._module_refs_dict


class ForwardTracker(TrackerBase):
    """
    A tracker for monitoring the forward pass of specific modules in a model.

    This tracker automatically appends the outputs of specified modules during a single forward
    pass to the `data` attribute as a list. The length of this list corresponds to the number of
    specified modules that were passed through during the forward pass.

    Args:
        tracker_fn (Callable):
            A callback function that is invoked after the default behavior of
            recording the output. This function does not modify the input or
            output of the module. The function should have the following signature::

                tracker_fn(module, args, output, module_name)

            where `args` are the inputs to the module's forward method. If the
            module has a single input, `args` will be that input. If the module
            has multiple inputs, `args` will be a tuple of those inputs.
    """

    def __init__(self, tracker_fn: Optional[Callable] = None):
        super().__init__()
        self._tracker_fn = (
            tracker_fn if tracker_fn else lambda m, args, output, module_name: None
        )

    @property
    def data(self):
        if not hasattr(self, "_data"):
            raise RuntimeError(
                "Attempting to get data from the tracker that was not attached to any model."
            )
        return self._data["data"]

    def _register_tracker(self, module: nn.Module) -> RemovableHandle:
        @self._hook_wrapper
        def hook(m, args, output):
            self._data = self._data if hasattr(self, "_data") else {"data": [[]]}
            self._data["data"][-1].append(clone_to_device(output))

        return module.register_forward_hook(hook)


class GradTracker(TrackerBase):
    """
    A tracker for monitoring the gradients during the backward pass of specific modules in a model.

    This tracker automatically appends the gradients of the specified modules during the backward pass
    to the `data` attribute as a list. The length of this list corresponds to the number of modules whose
    gradients were tracked during the backward pass.

    Args:
        tracker_fn (Callable, optional):
            A callback function that is invoked after recording the gradients. This function does not modify
            the gradients of the module. The function should have the following signature::

                tracker_fn(module, grad_input, grad_output, module_name)

            where `grad_input` and `grad_output` are the gradients of the inputs and outputs to the module's
            backward method, respectively. If the module has a single input or output, `grad_input` or
            `grad_output` will be that input/output. If the module has multiple inputs/outputs, the respective
            arguments will be tuples.
    """

    def __init__(self, tracker_fn: Optional[Callable] = None):
        super().__init__()
        self._tracker_fn = (
            tracker_fn if tracker_fn else lambda m, gi, go, module_name: None
        )

    @property
    def grad_inputs(self):
        if not hasattr(self, "_data"):
            raise RuntimeError(
                "Attempting to get grad inputs from the tracker that was not attached to any model."
            )
        return self._data["grad_inputs"]

    @property
    def grad_outputs(self):
        if not hasattr(self, "_data"):
            raise RuntimeError(
                "Attempting to get grad outputs from the tracker that was not attached to any model."
            )
        return self._data["grad_outputs"]

    def _register_tracker(self, module: nn.Module) -> RemovableHandle:
        @self._hook_wrapper
        def hook(m, grad_input, grad_output):
            self._data = (
                self._data
                if hasattr(self, "_data")
                else {"grad_inputs": [[]], "grad_outputs": [[]]}
            )

            self._data["grad_inputs"][-1].append(clone_to_device(grad_input))
            self._data["grad_outputs"][-1].append(clone_to_device(grad_output))

        return module.register_backward_hook(hook)
