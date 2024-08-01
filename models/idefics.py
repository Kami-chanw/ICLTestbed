from typing import List

from PIL import Image
import torch
import os

from models.model_base import ModelBase

from contextlib import suppress
from transformers import BitsAndBytesConfig
from transformers import (
    BatchFeature,
    IdeficsForVisionText2Text,
    IdeficsProcessor,
    AutoProcessor,
    AutoTokenizer,
)


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
        ).to(device)

    @property
    def model_name(self):
        return "idefics-9b"

    @property
    def default_prompt_template(self):
        # fmt: off
        template = (
            "{% if messages[0]['role'] == 'instruction' %}"
                "Instruction: {{ messages[0]['content'] }}\n"
                "{% set messages = messages[1:] %}"
            "{% endif %}"
            "{% for message in messages %}"
                "Question:" 
                "{% for line in message['query'] %}"
                    "{% if line['type'] == 'text' %}"
                        "{{ line['text'] }}"
                    "{% elif line['type'] == 'image' %}"
                        "{{- '<image>' }}"
                    "{% endif %}"
                "{% endfor %}"
                "<end_of_utterance>\n"
                "{% if 'answer' in message %}"
                    "Short answer: " 
                    "{% for line in message['answer'] %}"
                        "{% if line['type'] == 'text' %}"
                            "{{ line['text'] }}"
                        "{% elif line['type'] == 'image' %}"
                            "{{- '<image>' }}"
                        "{% endif %}"
                    "{% endfor %}"
                    "<end_of_utterance>\n"
                "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
                "Short answer: " 
            "{% endif %}"
        )
        # fmt: on

    def generate(self, texts, images, **kwargs):
        texts = self.apply_prompt_template(texts, add_generation_prompt=True)
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
            inputs.append(result)
        return self._generate(
            {"prompts": inputs, "padding": True, "return_tensors": "pt"}, kwargs
        )
