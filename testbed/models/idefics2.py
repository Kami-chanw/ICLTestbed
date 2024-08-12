from testbed.models.model_base import ModelBase


import warnings
from transformers import (
    Idefics2ForConditionalGeneration,
    Idefics2Processor,
)


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
            return template.replace("<end_of_utterance>", "\n")
        return template

    def generate(self, text, images, **kwargs):
        text = self.apply_prompt_template(text)
        kwargs.pop("return_inputs", False)
        return self._generate(
            {"text": text, "images": images, "padding": True, "return_tensors": "pt"},
            kwargs,
        )
