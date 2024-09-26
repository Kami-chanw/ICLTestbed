import re
from transformers import (
    IdeficsForVisionText2Text,
    IdeficsProcessor,
)

from testbed.models.model_base import HookType, ModelBase


class Idefics(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=IdeficsProcessor,
        model_class=IdeficsForVisionText2Text,
        dtype=None,
        device=None,
    ):
        super().__init__(device)

        self.processor = processor_class.from_pretrained(
            model_root,
            torch_dtype=dtype,
        )

        self.model = model_class.from_pretrained(
            model_root,
            torch_dtype=dtype,
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
            module_name_or_type = r"model\." + pattern_prefix + r"layers\.\d+$"

        return super()._register_hook(
            register_fn_name,
            module_name_or_type,
            hook,
            use_regex=kwargs.get("use_regex", False) or pattern_prefix is not None,
            **kwargs,
        )

    @property
    def default_prompt_template(self):
        # see https://arxiv.org/pdf/2306.16527
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
                        "{% if loop.last %}"
                            "{% if message['role'] == 'answer' or message['role'] == 'caption' %}"
                                "\n\n"
                            "{% else %}"
                                " "
                            "{%+ endif %}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on

    def process_input(self, texts, images, padding=True, return_tensors="pt", **kwargs):
        if isinstance(texts[0], dict) or (
            isinstance(texts[0], list) and isinstance(texts[0][0], dict)
        ):
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

        return self.processor(
            prompts=inputs, padding=padding, return_tensors=return_tensors, **kwargs
        )
