import warnings
from transformers import (
    Idefics2ForConditionalGeneration,
    Idefics2Processor,
)

from testbed.models.model_base import HookType, ModelBase


class Idefics2(ModelBase):
    BASE_MODEL_NAME = "idefics2-8b-base"
    FINE_TUNED_MODEL_NAME = "idefics2-8b"

    def __init__(
        self,
        model_root,
        processor_class=Idefics2Processor,
        model_class=Idefics2ForConditionalGeneration,
        dtype=None,
        device=None,
    ):
        super().__init__(device)

        if self.BASE_MODEL_NAME in model_root:
            self._model_name = self.BASE_MODEL_NAME
        elif self.FINE_TUNED_MODEL_NAME in model_root:
            self._model_name = self.FINE_TUNED_MODEL_NAME
        else:
            warnings.warn(
                "The model type cannot be detected automatically in `model_root`, which may lead to unexpected behaviors."
            )
            self._model_name = None

        self.processor = processor_class.from_pretrained(
            model_root,
            torch_dtype=dtype,
            do_image_splitting=False,
            chat_template=self.default_prompt_template,
        )

        self.model = model_class.from_pretrained(
            model_root,
            torch_dtype=dtype,
            device_map=device,
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
        if pattern_prefix is not None:
            module_name_or_type = pattern_prefix + r"layers\.\d+$"

        return super()._register_hook(
            register_fn_name,
            module_name_or_type,
            hook,
            use_regex=kwargs.get("use_regex", False) or pattern_prefix is not None,
            **kwargs
        )

    @property
    def default_prompt_template(self):
        # adopt idefics1 prompt template, see https://arxiv.org/pdf/2306.16527
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
                        "{% if loop.last %}"
                            "{% if message['role'] == 'answer' or message['role'] == 'caption' %}"
                                "<end_of_utterance>\n"
                            "{% else %}"
                                " "
                            "{%+ endif %}"
                        "{% endif %}"
                    "{% endfor %}"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on
        if self.model_name == self.BASE_MODEL_NAME:
            # base model doesn't have <end_of_utterance> token
            return template.replace("<end_of_utterance>", "\n")
        return template

    def process_input(self, texts, images, padding=True, return_tensors="pt", **kwargs):
        if isinstance(texts[0], dict) or (
            isinstance(texts[0], list) and isinstance(texts[0][0], dict)
        ):
            texts = self.apply_prompt_template(texts)
        return self.processor(
            text=texts,
            images=images,
            padding=padding,
            return_tensors=return_tensors,
            **kwargs
        )
