import os
import re
import warnings
from typing import Any, Dict, List, Union
from PIL.Image import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
)

from testbed.models.model_base import ModelBase


class Idefics2(ModelBase):
    def __init__(
        self,
        model_root,
        processor_class=AutoProcessor,
        model_class=AutoModelForVision2Seq,
        processor_args=None,
        model_args=None,
        **common_args,
    ):
        self._model_name = os.path.basename(model_root).lower()
        if not re.fullmatch(r"^idefics2-\d+b[a-zA-Z-]*$", self._model_name):
            warnings.warn(
                "The model type cannot be detected automatically in `model_root`, which may lead to unexpected behaviors."
            )
            self._model_name = None

        processor_args = (
            processor_args
            if processor_args
            else dict(
                chat_template=self.default_prompt_template, do_image_splitting=False
            )
        )

        super().__init__(
            model_root=model_root,
            processor_class=processor_class,
            model_class=model_class,
            processor_args=processor_args,
            model_args=model_args,
            **common_args,
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
            "{% set last_role = messages[0]['role'] %}"
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
                        "{% if not loop.last %}"
                            " "
                        "{%+ endif %}"
                    "{% endfor %}"
                    "{% set is_end_of_round = loop.nextitem is not defined or loop.nextitem['role'] == last_role %}"
                    "{% if is_end_of_round %}"
                        "<end_of_utterance\n>"
                    "{% else %}"
                        " "
                    "{%+ endif %}"
                "{% endif %}"
            "{% endfor %}"
        )
        # fmt: on
        if "base" in self._model_name:
            # base model doesn't have <end_of_utterance> token
            return template.replace("<end_of_utterance>", "\n")
        return template

    def process_input(
        self,
        text: Union[
            List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]
        ],
        images: Union[List[Image], List[List[Image]]],
        prompt_template: str = None,
        **kwargs,
    ):
        """
        Processes text and image inputs for the model.

        Args:
            text (Union[List[Union[str, Dict[str, Any]]], List[List[Union[str, Dict[str, Any]]]]]):
                A list of texts or a list of lists of texts. For unbatched input, this should be a single-level list
                where each item is either a string or a dictionary. For batched input, this should be a nested list
                (list of lists) where each inner list represents a batch of texts. Dictionaries can follow the
                transformers' conversation format, with keys like "role" and "content".

            images (Union[List[Image], List[List[Image]]]):
                A list of images or a list of lists of images. For unbatched input, this should be a single-level list
                of images. For batched input, this should be a nested list where each inner list represents a batch of images.
                Each image should be an instance of the `Image` class.

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
            images=images,
            padding=kwargs.pop("padding", True),
            return_tensors=kwargs.pop("return_tensors", "pt"),
            **kwargs,
        )
