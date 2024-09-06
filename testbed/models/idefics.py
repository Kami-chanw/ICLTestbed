import inspect
import re
from typing import Callable, List
import warnings
import torch
from torch.utils.hooks import RemovableHandle
from transformers import (
    IdeficsForVisionText2Text,
    IdeficsProcessor,
)

from testbed.models.model_base import HookType, ModelBase


class Idefics(ModelBase):
    def __init__(self, model_root, precision, device=None):
        super().__init__(precision, device)

        self.processor = IdeficsProcessor.from_pretrained(
            model_root,
            torch_dtype=self.cast_dtype,
            local_files_only=True,
        )

        self.model = IdeficsForVisionText2Text.from_pretrained(
            model_root,
            torch_dtype=self.cast_dtype,
            local_files_only=True,
            device_map=device,
        )

    @property
    def model_name(self):
        return "idefics-9b"

    def _register_hook(self, register_fn_name, module_name_or_type, hook, **kwargs):
        pattern_prefix = None
        if module_name_or_type == HookType.TEXT_MODEL_LAYER:
            pattern_prefix = ""
        elif module_name_or_type == HookType.VISION_MODEL_LAYER:
            pattern_prefix = r"vision_model\.encoder\."
        if pattern_prefix is not None:
            signature = inspect.signature(getattr(torch.nn.Module, register_fn_name))
            # Remove 'always_call' from kwargs if it's not supported in this version
            if "always_call" in kwargs and "always_call" not in signature.parameters:
                kwargs.pop("always_call")
            return [
                getattr(module, register_fn_name)(hook, **kwargs)
                for name, module in self.model.named_modules()
                if re.search(r"model\." + pattern_prefix + r"layers\.\d+$", name)
            ]
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
                    "{{ message['role'].capitalize() }}"
                    "{% if not 'content' in message or message['content'][0]['type'] == 'image' %}"
                        "{{':'}}"
                    "{% else %}"
                        "{{': '}}"
                    "{% endif %}" 
                "{% endif %}"
                "{% if 'content' in message %}"
                    "{% for line in message['content'] %}"
                        "{% if line['type'] == 'text' %}"
                            "{{ line['text'] }}"
                        "{% elif line['type'] == 'image' %}"
                            "{{- '<image>' }}"
                        "{% endif %}"
                    "{% endfor %}"
                "\n\n"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on

    def generate(self, texts, images, **kwargs):
        texts = self.apply_prompt_template(texts)
        inputs = []
        for i, (text, image_list) in enumerate(zip(texts, images)):
            text = text.split("<image>")
            result = []
            if len(text) - 1 != len(image_list):
                raise ValueError(
                    f"In the {i}-th input, the number of images does not match the number of image tokens in the text."
                )
            for seg, image in zip(text, image_list):
                if seg != "":
                    result.append(seg)
                result.append(image)
            if text[-1] != "":  # the last question without answer
                result.append(text[-1])
            inputs.append(result)

        ex_args = self._extract_extra_generate_args(kwargs)
        return self._generate(
            {"prompts": inputs, "padding": True, "return_tensors": "pt"},
            kwargs,
            **ex_args,
        )
