import os
import re
import warnings
from typing import Any, Dict, List, Union
from PIL.Image import Image
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from testbed.models.model_base import HookType, ModelBase


class Mistral(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoTokenizer,
        model_class=AutoModelForCausalLM,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        self._model_name = os.path.basename(model_root).lower()
        if not re.fullmatch(r"^mistral-\d+b[a-zA-Z0-9-]*$", self._model_name):
            warnings.warn(
                "The model type cannot be detected automatically in `model_root`, which may lead to unexpected behaviors."
            )
            self._model_name = None

        processor_args = (
            processor_args
            if processor_args
            else dict(chat_template=self.default_prompt_template)
        )

        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
        )

    def _register_hook(self, register_fn_name, module_name_or_type, hook, **kwargs):
        if module_name_or_type == HookType.TEXT_MODEL_LAYER:
            module_name_or_type = r"model\.layers\.\d+$"
        elif isinstance(module_name_or_type, HookType):
            raise ValueError(
                f"{__class__.__name__} doesn't support hook type of {module_name_or_type.name}"
            )

        return super()._register_hook(
            register_fn_name, module_name_or_type, hook, **kwargs
        )

    @property
    def default_prompt_template(self):
        # fmt: off
        return (
            "{% if messages[0]['role'] == 'instruction' %}"
                "Instruction: {{ messages[0]['content'] }}\n"
                "{% set messages = messages[1:] %}"
            "{% endif %}"
            "{% for message in messages %}"
                "{% if message['role'] != '' %}"
                    "{{ message['role'].capitalize() }}: "
                "{%+ endif %}"
                "{% if 'content' in message %}"
                    "{% for line in message['content'] %}"
                        "{% if line['type'] == 'text' %}"
                            "{{ line['text'] }}"
                        "{% endif %}"
                        "{% if loop.last %}"
                            "\n\n"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on

    def process_input(
        self,
        text: Union[
            List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]
        ],
        prompt_template: str = None,
        **kwargs,
    ):
        """
        Processes text inputs for the model.

        Args:
            text (Union[List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]]):
                A list of texts or a list of lists of texts. For unbatched input, this should be a single-level list
                where each item is either a string or a dictionary. For batched input, this should be a nested list
                (list of lists) where each inner list represents a batch of texts. Dictionaries can follow the
                transformers' conversation format, with keys like "role" and "content".

            prompt_template (str, optional):
                An optional template string used to format the input texts if they are provided as dictionaries.

            **kwargs:
                Additional keyword arguments passed to the `processor`.

        Returns:
            The output of the `processor` function, which is the processed input ready for the model.
        """
        if isinstance(text[0], dict) or (
            isinstance(text[0], list) and isinstance(text[0][0], dict)
        ):
            text = self.apply_prompt_template(text, prompt_template=prompt_template)
        
        return self.processor(
            text=text,
            padding=kwargs.pop("padding", True),
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )
