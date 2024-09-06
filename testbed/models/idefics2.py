import inspect
import re
from typing import Callable, List
import warnings
import torch
from torch.utils.hooks import RemovableHandle
from transformers import (
    Idefics2ForConditionalGeneration,
    Idefics2Processor,
)

from testbed.models.model_base import HookType, ModelBase


class Idefics2(ModelBase):
    BASE_MODEL_NAME = "idefics2-8b-base"
    FINE_TUNED_MODEL_NAME = "idefics2-8b"

    def __init__(
        self, model_root, precision="bf16", device=None, quantization_config=None
    ):
        super().__init__(precision, device)

        if self.BASE_MODEL_NAME in model_root:
            self._model_name = self.BASE_MODEL_NAME
        elif self.FINE_TUNED_MODEL_NAME in model_root:
            self._model_name = self.FINE_TUNED_MODEL_NAME
        else:
            warnings.warn(
                "The model type cannot be detected automatically in `model_root`, which may lead to unexpected behaviors."
            )
            self._model_name = None

        self.processor = Idefics2Processor.from_pretrained(
            model_root,
            torch_dtype=self.cast_dtype,
            local_files_only=True,
            do_image_splitting=False,
            chat_template=self.default_prompt_template,
        )

        self.model = Idefics2ForConditionalGeneration.from_pretrained(
            model_root,
            torch_dtype=self.cast_dtype,
            local_files_only=True,
            device_map=device,
            quantization_config=quantization_config,
        )

    @property
    def model_name(self):
        return self._model_name

    def _register_hook(self, register_fn_name, module_name_or_type, hook, **kwargs):
        pattern_prefix = None
        if module_name_or_type == HookType.TEXT_MODEL_LAYER:
            pattern_prefix = r"text_model\."
        elif module_name_or_type == HookType.VISION_MODEL_LAYER:
            pattern_prefix = r"vision_model\.encoder\."
        if pattern_prefix:
            signature = inspect.signature(getattr(torch.nn.Module, register_fn_name))
            # Remove 'always_call' from kwargs if it's not supported in this version
            if "always_call" in kwargs and "always_call" not in signature.parameters:
                kwargs.pop("always_call")
            return [
                getattr(module, register_fn_name)(hook, **kwargs)
                for name, module in self.model.named_modules()
                if re.search(pattern_prefix + r"layers\.\d+$", name)
            ]
        return super()._register_hook(
            register_fn_name, module_name_or_type, hook, **kwargs
        )

    @property
    def default_prompt_template(self):
        # fmt: off
        template = (
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
                "<end_of_utterance>\n"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on
        if self.model_name == self.BASE_MODEL_NAME:
            # base model doesn't have <end_of_utterance> token
            return template.replace("<end_of_utterance>", "\n")
        return template

    def generate(self, text, images, **kwargs):
        text = self.apply_prompt_template(text)
        ex_args = self._extract_extra_generate_args(kwargs)
        return self._generate(
            {"text": text, "images": images, "padding": True, "return_tensors": "pt"},
            kwargs,
            **ex_args,
        )
